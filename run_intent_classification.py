import os
import argparse
import platform
from glob import glob

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

class IntentClassification(pl.LightningModule):
    def __init__(self,
                 domain,
                 text_reader,
                 num_intents,
                 learning_rate: float=2e-5,
                 num_samples: int=30
                 ):
        super().__init__()
        self.save_hyperparameters()

        # prepare text reader
        from utils import get_text_reader
        text_reader = get_text_reader(self.hparams.domain, self.hparams.text_reader)
        self.text_reader = text_reader

        # Dimension Reduction : [CLS]768 --> 500 --> 300 --> 100
        self.fc1 = nn.Linear(768, 500)
        self.fc2 = nn.Linear(500, 300)
        self.fc3 = nn.Linear(300, 100)

        # to intent class
        self.to_class = nn.Linear(100, num_intents)

        # number of dropout examples for generating perturbation
        self.num_samples = num_samples

    def forward(self, input_ids, token_type_ids, attention_mask):
        # v_t is [CLS] vector
        _, v_t = self.text_reader(
            input_ids=input_ids.long(),
            token_type_ids=token_type_ids.long(),
            attention_mask=attention_mask.float()
        )

        # Dimension Reduction : [CLS] vector --> 100 dim
        x = torch.tanh(self.fc1(v_t))
        x = torch.tanh(self.fc2(x))
        target_vt = torch.tanh(self.fc3(x)) # we use these vectors !

        # Logit Vector --> Dimension : num labels
        logits = self.to_class(target_vt)

        return target_vt, logits

    def training_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, intent_id = batch

        _, logits = self(input_ids, token_type_ids, attention_mask)

        loss = F.cross_entropy(logits, intent_id.long())

        result = {"loss": loss}
        return result

    def validation_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, intent_id = batch

        _, logits = self(input_ids, token_type_ids, attention_mask)

        loss = F.cross_entropy(logits, intent_id.long())
        preds = torch.argmax(logits, dim=1)

        labels = intent_id
        result = {"loss": loss, "preds": preds, "labels": labels}
        return result

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs])
        loss = torch.stack([x["loss"] for x in outputs]).mean()

        correct_count = torch.sum(labels == preds)
        val_acc = correct_count.float() / float(len(labels))

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx):
        # dataloader_idx : 0 --> pure test dataloader
        # dataloader_idx : 1 --> train dataloader for generating vector movement
        # dataloader_idx : 2 --> test dataloader for generating vector movement

        input_ids, token_type_ids, attention_mask, intent_id = batch

        if dataloader_idx == 0: # only testing mode
            _, logits = self(input_ids, token_type_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            labels = intent_id
            result = {"preds": preds, "labels": labels}

        else: # dumping mode
            # duplicate target vector per num_samples
            input_ids = torch.repeat_interleave(input_ids, self.num_samples, dim=0)
            token_type_ids = torch.repeat_interleave(token_type_ids, self.num_samples, dim=0)
            attention_mask = torch.repeat_interleave(attention_mask, self.num_samples, dim=0)

            self.train() # activate to dropout for generating perturbated vector embedding !
            target_vectors, _ = self(input_ids, token_type_ids, attention_mask) # target vectors are vector movement with dropout !

            if dataloader_idx == 1: # train-set
                result = {"train_vt" : target_vectors}
            else: # test-set
                result = {"test_vt": target_vectors}

        return result

    def test_epoch_end(self, outputs):
        # outputs[0] --> pure preds and labels, after testing
        # outputs[1] --> vector movement about train-set
        # outputs[2] --> dumping target vectors about test-set

        # measure metric --> accuracy
        test_output = outputs[0]
        dumping_output_about_trainset = outputs[1]
        dumping_output_about_testset = outputs[2]

        preds = torch.cat([x["preds"] for x in test_output])
        labels = torch.cat([x["labels"] for x in test_output])

        correct_count = torch.sum(labels == preds)
        test_acc = correct_count.float() / float(len(labels))

        self.log("test_acc", test_acc, prog_bar=True)

        # dump vector movements about train-set and test-set
        self.dump_vectors(dumping_output_about_trainset, "train")
        self.dump_vectors(dumping_output_about_testset, "test")

        return test_acc

    def dump_vectors(self, outputs, flag):
        vt_fn = os.path.join(self.trainer.callbacks[1].dirpath, '{}.v_t.{}'.format(flag, self.num_samples))

        vt = torch.stack([x["{}_vt".format(flag)] for x in outputs]) # [num_examples, num_samples, target_vector_dim]
        vt = vt.data.cpu().numpy()

        np.save(vt_fn, vt)
        print("\n[{}] vt file is dumped at {} - {}".format(flag, vt_fn, vt.shape))

    def configure_optimizers(self):
        from transformers import AdamW

        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=2e-5)
        return parser


def main():
    pl.seed_everything(42) # set seed

    # Argument Setting -------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()

    # domain ---------------------------------------------------------------------------------------
    parser.add_argument("--domain", help="What domain do you want?", default="weather")

    # mode specific --------------------------------------------------------------------------------
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to train intent classifier.")
    parser.add_argument("--do_test_and_dump", action='store_true',
                        help="Whether to test and dump target vector.")

    # model specific -------------------------------------------------------------------------------
    parser.add_argument("--text_reader", help="lstm, bert, xlnet, others, ...", default="bert")  # bert fixed for now

    # experiment settings --------------------------------------------------------------------------
    parser.add_argument("--batch_size", help="batch_size", default=50)
    parser.add_argument("--max_seq_length", default=40, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded. \n"
                             "We recommend that you set it to 128(en) or 40(kor)."
                        )

    # params for generating perturbated vector movements -------------------------------------------
    parser.add_argument("--num_samples", help="number of dropout example for generating perturbation", type=int, default=50)

    parser = pl.Trainer.add_argparse_args(parser)
    parser = IntentClassification.add_model_specific_args(parser)
    args = parser.parse_args()
    # ------------------------------------------------------------------------------------------------------------------

    # Dataset ----------------------------------------------------------------------------------------------------------
    from dataset import Intent_Classification_Data_Module
    dm = Intent_Classification_Data_Module(args.domain, args.text_reader, args.max_seq_length, args.batch_size)
    dm.prepare_data()
    # ------------------------------------------------------------------------------------------------------------------

    # Model Checkpoint -------------------------------------------------------------------------------------------------
    from pytorch_lightning.callbacks import ModelCheckpoint
    text_reader_model_name = '{}'.format(args.text_reader)
    model_folder = './model/{}/{}'.format(args.domain, text_reader_model_name)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          dirpath=model_folder,
                                          filename='{epoch:02d}-{val_loss:.2f}')
    # ------------------------------------------------------------------------------------------------------------------

    # Early Stopping ---------------------------------------------------------------------------------------------------
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        verbose=True
    )
    # ------------------------------------------------------------------------------------------------------------------

    # Trainer ----------------------------------------------------------------------------------------------------------
    trainer = pl.Trainer(
        gpus=args.gpus if platform.system() != 'Windows' else 1,  # <-- for dev. pc
        checkpoint_callback=checkpoint_callback,
        callbacks=[early_stop_callback]
    )
    # ------------------------------------------------------------------------------------------------------------------

    # Do train !
    if args.do_train:
        model = IntentClassification(args.domain, args.text_reader, dm.num_intents)
        trainer.fit(model, dm)

    # Do test and dump !
    if args.do_test_and_dump:
        model_files = glob(os.path.join(model_folder, '*.ckpt'))
        best_fn = model_files[-1]
        model = IntentClassification.load_from_checkpoint(best_fn)
        model.num_samples = args.num_samples
        trainer.test(model, test_dataloaders=[dm.test_dataloader(), dm.train_dataloader_for_dump(), dm.test_dataloader_for_dump()])

if __name__ == '__main__':
    main()
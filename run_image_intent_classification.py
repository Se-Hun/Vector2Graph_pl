import os
import argparse
import platform
from glob import glob

import torch
import torch.optim as optim
from torch.optim import Adam
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

class ImageIntentClassification(pl.LightningModule):
    def __init__(self,
                 domain,
                 image_reader,
                 num_intents,
                 learning_rate: float=3e-5,
                 ):
        super().__init__()
        self.save_hyperparameters()

        # prepare image reader
        from utils import get_image_reader
        image_reader = get_image_reader(self.hparams.image_reader, num_intents)
        self.image_reader = image_reader

    def forward(self, b_img_tensor, b_labels):
        logit_v_intent, last_hidden_state = self.image_reader(b_img_tensor, b_labels)

        return logit_v_intent, last_hidden_state

    def training_step(self, batch, batch_idx):
        b_img_tensor, b_labels = batch

        logits, _ = self(b_img_tensor, b_labels)

        loss = F.cross_entropy(logits, b_labels)

        result = {"loss": loss}
        return result

    def validation_step(self, batch, batch_idx):
        b_img_tensor, b_labels = batch

        logits, _ = self(b_img_tensor, b_labels)

        loss = F.cross_entropy(logits, b_labels)
        preds = torch.argmax(logits, dim=1)

        labels = b_labels
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

    def test_step(self, batch, batch_idx):
        b_img_tensor, b_labels = batch

        logits, _ = self(b_img_tensor, b_labels)

        preds = torch.argmax(logits, dim=1)

        labels = b_labels
        result = {"preds": preds, "labels": labels}
        return result

    def test_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs])

        correct_count = torch.sum(labels == preds)
        test_acc = correct_count.float() / float(len(labels))

        self.log("test_acc", test_acc, prog_bar=True)
        return test_acc

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
        optimizer = Adam(optimizer_grouped_parameters, lr=self.hparams.learning_rate)
        max_grad_norm = 1.0

        # scheduler
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        #                                                  'min',
        #                                                  factor=0.99,
        #                                                  verbose=True)
        # return optimizer, scheduler
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=3e-5)
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
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to evaluate on test set.")

    # model specific -------------------------------------------------------------------------------
    parser.add_argument("--image_reader", help="cnn, resnet, vgg, others, ...", default="cnn")  # cnn fixed for now

    # experiment settings --------------------------------------------------------------------------
    parser.add_argument("--batch_size", help="batch_size", default=50)

    # params about generated vector movements ------------------------------------------------------
    parser.add_argument("--num_samples", help="number of generated perturbation's samples", type=int, default=50)

    # params about converted representation --------------------------------------------------------
    parser.add_argument("--base_text_reader", default="bert",
                        help="when generating vector movement, what text reader was used ?") # fixed to bert in this version
    parser.add_argument("--rep_type", default="graph",
                        help="experiment type : graph, flatted_bar, circular_bar")  # fixed to graph in this version
    parser.add_argument("--need_edges", action='store_true',
                        help="experiment type : graph_with_edge, graph_wo_edge")
    parser.add_argument("--top_n", help='number of nodes about graph, others',
                        default=10)

    parser = pl.Trainer.add_argparse_args(parser)
    parser = ImageIntentClassification.add_model_specific_args(parser)
    args = parser.parse_args()
    # ------------------------------------------------------------------------------------------------------------------

    # Dataset ----------------------------------------------------------------------------------------------------------
    from dataset import Image_Intent_Classification_Data_Module
    need_edges = args.need_edges
    edge_flag = "with_edge" if need_edges else "wo_edge"
    data_dir = os.path.join("./", "images", args.domain, args.base_text_reader,
                            "{}_{}".format(args.rep_type, args.num_samples), "top_{}_{}".format(args.top_n, edge_flag))
    dm = Image_Intent_Classification_Data_Module(data_dir, args.batch_size)
    dm.prepare_data()
    # ------------------------------------------------------------------------------------------------------------------

    # Model Checkpoint -------------------------------------------------------------------------------------------------
    from pytorch_lightning.callbacks import ModelCheckpoint
    image_reader_model_name = '{}'.format(args.image_reader)
    model_folder = './model/{}/{}_{}/top_{}_{}/{}'.format(args.domain, args.rep_type, args.num_samples,
                                                          args.top_n, edge_flag, image_reader_model_name)
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
        model = ImageIntentClassification(args.domain, args.image_reader, dm.num_intents)
        trainer.fit(model, dm)

    # Do test and dump !
    if args.do_test:
        model_files = glob(os.path.join(model_folder, '*.ckpt'))
        best_fn = model_files[-1]
        model = ImageIntentClassification.load_from_checkpoint(best_fn)
        trainer.test(model, test_dataloaders=[dm.test_dataloader()])

if __name__ == '__main__':
    main()
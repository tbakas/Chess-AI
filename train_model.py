import pytorch_lightning.loggers
from torchmetrics.classification import MulticlassAccuracy
from create_model import ChessModel
import torch
import lightning
import numpy
import logging


logger = logging.getLogger(__name__)
max_epochs = 30
batch_size = 128
base_path = 'drive/MyDrive/'
train_perc = 0.8
val_perc = 0.1


class ChessLightning(lightning.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.val_top5 = MulticlassAccuracy(num_classes=1968, top_k=5, average='micro')
        self.test_top5 = MulticlassAccuracy(num_classes=1968, top_k=5, average='micro')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs)
        optim_dict = {'optimizer': optimizer,
                      'lr_schedule': {'scheduler': lr_scheduler, 'step': 'epoch', 'frequency': 1}}
        return optim_dict

    def training_step(self, batch, batch_idx):
        x_batch, t_batch, y_batch, m_batch = batch
        masked_logits = self.model(x_batch, t_batch).masked_fill(m_batch == 0, float('-inf'))
        loss = torch.nn.functional.cross_entropy(masked_logits, y_batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x_batch, t_batch, y_batch, m_batch = batch
        masked_logits = self.model(x_batch, t_batch).masked_fill(m_batch == 0, float('-inf'))
        loss = torch.nn.functional.cross_entropy(masked_logits, y_batch)
        self.val_top5(masked_logits, y_batch)
        self.log('val_loss', loss)
        self.log('val_acc', self.val_top5)

    def test_step(self, batch, batch_idx):
        x_batch, t_batch, y_batch, m_batch = batch
        masked_logits = self.model(x_batch, t_batch).masked_fill(m_batch == 0, float('-inf'))
        loss = torch.nn.functional.cross_entropy(masked_logits, y_batch)
        self.test_top5(masked_logits, y_batch)
        self.log('test_loss', loss)
        self.log('test_acc', self.test_top5)


class ChessDataset(torch.utils.data.Dataset):
    def __init__(self, x_path, t_path, y_path, mask_path):
        self.x = numpy.load(x_path, mmap_mode='r')
        self.t = numpy.load(t_path, mmap_mode='r')
        self.y = numpy.load(y_path, mmap_mode='r')
        self.packed_mask = numpy.load(mask_path, mmap_mode='r')

    def __len__(self):
        return self.t.shape[0]

    def __getitem__(self, item):
        x_tensor = torch.tensor(self.x[item], dtype=torch.long)
        t_tensor = torch.tensor(self.t[item], dtype=torch.float32)
        y_tensor = torch.tensor(self.y[item], dtype=torch.long)
        mask_tensor = torch.tensor(numpy.unpackbits(self.packed_mask[item], axis=0), dtype=torch.bool)
        return x_tensor, t_tensor, y_tensor, mask_tensor


if __name__ == '__main__':
    logging.basicConfig(filename='train_log.log',
                        format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f'The device used is {device}')
    chess_model = ChessModel().to(device)

    logger.info('Opening transformed_data.csv')
    dataset = ChessDataset(base_path + 'x.npy', base_path + 't.npy', base_path + 'y.npy', base_path + 'mask.npy')

    # split by game_id so that games don't appear in multiple data splits
    g = numpy.load(base_path + 'g.npy')
    indices = numpy.arange(len(dataset))
    n = m = -1
    count = 0
    for i in range(1, indices.shape[0]):
        count += 1
        if g[i] == g[i - 1]:
            continue
        elif n == -1 and (count / indices.shape[0]) >= train_perc:
            n = i
        elif m == -1 and (count / indices.shape[0]) >= train_perc + val_perc:
            m = i
            break
    del g

    train_dataset = torch.utils.data.Subset(dataset, indices[:n])
    val_dataset = torch.utils.data.Subset(dataset, indices[n:m])
    test_dataset = torch.utils.data.Subset(dataset, indices[m:])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    chess_lightning = ChessLightning(chess_model)
    checkpoint_callback = lightning.pytorch.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1,
                                                                      filename="{epoch}-{val_loss:.3f}",
                                                                      dirpath=base_path)

    logger_lightning = pytorch_lightning.loggers.CSVLogger(save_dir=base_path)
    trainer = lightning.Trainer(max_epochs=max_epochs, callbacks=[checkpoint_callback], logger=logger_lightning)

    logger.info('Beginning the training loop...')
    trainer.fit(chess_lightning, train_loader, val_loader)
    logger.info('Training completed.')

    del train_loader, train_dataset, val_dataset, val_loader

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    logger.info('Beginning testing.')
    trainer.test(chess_lightning, test_loader, ckpt_path='best')

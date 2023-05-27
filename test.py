import util
import argparse
import time
from model import *
from engine import trainer


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--data', type=str, default='data/', help='data path')
parser.add_argument('--num_nodes', type=int, default=207, help='node num')
parser.add_argument('--seq_length', type=int, default=12, help='')
parser.add_argument('--nhid', type=int, default=64, help='')
parser.add_argument('--edim', type=int, default=32, help='embedding dim')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.15, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--checkpoint', type=str, default='garage/', help='')

args = parser.parse_args()


def main():
    device = torch.device(args.device)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    engine = trainer(scaler, args.edim, args.seq_length, args.nhid, args.dropout, args.learning_rate,
                     args.weight_decay, args.device, args.num_nodes, args.batch_size)
    model = engine.model
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    print('model load successfully!')

    outputs = []
    realy = []
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        testy = torch.Tensor(y).to(device)
        testy = testy.transpose(1, 3)[:, :1, :, :]
        with torch.no_grad():
            preds = model(testx).transpose(1, 3)
        outputs.append(preds)
        realy.append(testy)

    yhat = torch.cat(outputs, dim=0)
    yhat = scaler.inverse_transform(yhat)
    realy = torch.cat(realy, dim=0)

    amae = []
    amape = []
    armse = []
    for i in range(args.seq_length):
        pred = yhat[..., i]
        real = realy[..., i]
        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(*util.metric(yhat, realy)))


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))

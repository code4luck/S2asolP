from argparse import ArgumentParser
from src.wrapperdataset import *
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
from infer_proxy import ProxyModel
import torch, warnings, os
import torch.nn.functional as F

warnings.filterwarnings("ignore")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="deepsol", choices=DATASETS)
    parser.add_argument("--data_path", nargs="+", type=str, default=None)
    parser.add_argument("--bio_feature_paths", nargs="+", type=str, default=None)
    parser.add_argument("--split_method", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--model_path", type=str, default="saprot_pdb")
    parser.add_argument("--tokenizer", type=str, default="saprot_pdb")

    # model
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--pooling_head", type=str, default="attention1d")
    parser.add_argument("--hidden_size", type=int, default=1280)
    parser.add_argument("--proj_dim", type=int, default=1280)
    parser.add_argument("--drop_rate", type=float, default=0.1)
    parser.add_argument("--seq_max_length", type=int, default=1200)

    # Trainer
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    seed_everything(args.seed)
    return args


@torch.no_grad()
def inference(model, dataset=None, args=None):
    device = args.device
    model.eval()
    model = model.to(device)

    for idx, batch in enumerate(dataset):
        model.test(batch, device)

    print(
        "================================predict_result================================="
    )
    data = [[i, j] for i, j in zip(model.predict_res[0], model.predict_res[1])]
    data = torch.tensor(data, dtype=torch.float32)
    prob = F.softmax(data, dim=-1)
    p0 = prob.numpy()[:, 0]
    p1 = prob.numpy()[:, 1]
    df = pd.DataFrame(
        {
            "true_labels": model.true_labels,
            "pred_insolubility": p0,
            "pred_solubility": p1,
        }
    )
    df.to_csv("./predict_result.csv", index=False)
    print(
        "================================end prediction================================="
    )


if __name__ == "__main__":
    args = parse_args()
    dataset = WrapperData(args)
    proxy_model = ProxyModel(
        args=args,
    )
    proxy_model.load_state_dict(
        torch.load("checkpoints/s2asolp_checkpoint.pt", map_location="cpu")
    )

    test_loader = dataset.get_testloader()
    inference(proxy_model, test_loader, args)

import sys
sys.path.append("/workspace/data/CARLA-ICTS")
import logging
import os
import time
import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from datetime import datetime as dt
from ped_path_predictor.new_util import getDataloaders, singleDatasets
from ped_path_predictor.autobots.AutoBots.models.autobot_ego import AutoBotEgo
from ped_path_predictor.autobots.AutoBots.utils.train_helpers import nll_loss_multimodes


path_int = "./ped_path_predictor/data/new_car/all_int.npy"
path_non_int = "./ped_path_predictor/data/new_car/all_non_int.npy"

path_int_car = "./ped_path_predictor/data/new_car/all_int_car.npy"
path_non_int_car = "./ped_path_predictor/data/new_car/all_non_int_car.npy"


n_obs = 15
n_pred = 20
batch_size = 512
lr = 0.001

kl_weight = 20.0
entropy_weight = 40.0

epoch_limit = 1000


class AutoBotWrapperCog:

    def __init__(self, path=None):
        start_time_str = dt.today().strftime("%Y-%m-%d_%H-%M-%S")
        obs_str = f'obs{n_obs}_pred{n_pred}'
        self.base_path = f'./_out/{self.__class__.__name__}/{obs_str}/{start_time_str}'
        os.makedirs(self.base_path, exist_ok=True)

        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.INFO,
                            filename=f'{self.base_path}/train.log',
                            format='%(asctime)s %(name)s: %(levelname)-8s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

        self.logger.info(f"Model: AutoBotEgo Cognitive as Agent")
        self.logger.info(f"Observation Frames: {n_obs}")
        self.logger.info(f"Prediction Frames: {n_pred}")
        self.logger.info(f"Batch Size: {batch_size}")
        self.logger.info(f"Initital Learning Rate: {lr}")



        self.model = AutoBotEgo(
            k_attr=2,
            d_k=128,
            _M=2,
            c=1,
            T=n_pred,
            L_enc=1,
            dropout=0.0,
            num_heads=16,
            L_dec=1,
            tx_hidden_size=384,
            use_map_img=False,
            use_map_lanes=False).cuda()

        self.optimiser = optim.Adam(self.model.parameters(), lr=lr, eps=1e-4)
        self.optimiser_scheduler = MultiStepLR(self.optimiser, milestones=[5, 10, 15, 20], gamma=0.5,
                                               verbose=True)

        if path is not None:
            self.model.load_state_dict(torch.load(path))
        #
        # self.train_loader, self.test_loader, self.val_loader = getDataloaders(path_int, path_non_int, path_int_car,
        #                                                                       path_non_int_car, n_obs, n_pred,
        #                                                                       batch_size=batch_size)
        #
        # self.logger.info(f"Train-Batches {len(self.train_loader)}")
        # self.logger.info(f"Test-Batches {len(self.test_loader)}")

    def transform(self, x, y):
        ego_in = x[:, :, :2]
        agent_in = x[:, :, 4:]

        cf_as_agent = x[:, :, 2:4]

        ego_out = y.cuda()

        map_lanes = torch.zeros((batch_size, 1, 1)).cuda()

        ex_mask = torch.ones((ego_in.shape[0], ego_in.shape[1], 1)).float()
        ego_in = torch.concatenate((ego_in, ex_mask), dim=-1).cuda()

        agents_in = torch.cat((torch.concatenate((agent_in, ex_mask), dim=-1).unsqueeze(-2),
                               torch.concatenate((cf_as_agent, ex_mask), dim=-1).unsqueeze(-2)), dim=-2).cuda()

        return ego_in, agents_in, map_lanes, ego_out

    def get_single_prediction(self, x, vehicle_past):
        # x_traj = x[:,0:2]
        # x_cf = x[:,2:]

        ego_in = x[ :, :2]
        agent_in = vehicle_past

        cf_as_agent = x[ :, 2:4]

        ego_in = torch.tensor(ego_in).float()
        agent_in = torch.tensor(agent_in).float()
        cf_as_agent = torch.tensor(cf_as_agent).float()

        map_lanes = torch.zeros((batch_size, 1, 1)).cuda()


        ex_mask = torch.ones((ego_in.shape[0], ego_in.shape[1])).float()
        ego_in = torch.concatenate((ego_in, ex_mask), dim=-1).cuda()
        agents_in = torch.cat((torch.concatenate((agent_in, ex_mask), dim=-1).unsqueeze(-2),
                               torch.concatenate((cf_as_agent, ex_mask), dim=-1).unsqueeze(-2)), dim=-2).cuda()

        ego_in = ego_in.unsqueeze(0)
        agents_in = agents_in.unsqueeze(0)
        map_lanes = map_lanes.unsqueeze(0)

        with torch.no_grad():
            pred_obs, mode_probs = self.model(ego_in, agents_in, map_lanes)
        path = pred_obs.cpu().squeeze().numpy()
        return path

    def _compute_ego_errors(self, ego_preds, ego_gt, ego_in=None):
        with torch.no_grad():
            ego_gt = ego_gt.transpose(0, 1).unsqueeze(0)
            ade_losses = torch.mean(torch.norm(ego_preds[:, :, :, :2] - ego_gt[:, :, :, :2], 2, dim=-1),
                                    dim=1).transpose(0,
                                                     1).cpu().numpy()
            fde_losses = torch.norm(ego_preds[:, -1, :, :2] - ego_gt[:, -1, :, :2], 2, dim=-1).transpose(0,
                                                                                                         1).cpu().numpy()

            a, f = torch.square(ego_preds[:, :, :, :2] - ego_gt[:, :, :, :2]).sum(-1).sqrt().sum().item(), torch.square(
                (ego_preds[:, -1:, :, :2] - ego_gt[:, -1:, :, :2])).sum(-1).sqrt().sum().item()

        return ade_losses, fde_losses, a, f

    def train(self):

        # eval variables
        best_eval = np.Inf
        best_eval_fde = np.Inf
        last_best_epoch = 0

        for epoch in range(0, 1000):
            print(f'Epoch {epoch}')
            did_epoch_better = False

            self.model.train()

            t_before = time.time()
            for i, (x, y) in enumerate(self.train_loader):
                print(f'\rBatch {i}/{len(self.train_loader)}', end='')

                ego_in, agents_in, map_lanes, ego_out = self.transform(x, y)

                pred_obs, mode_probs = self.model(ego_in, agents_in, map_lanes)

                nll_loss, kl_loss, post_entropy, ade_fde_loss = nll_loss_multimodes(pred_obs, ego_out[:, :, :2],
                                                                                    mode_probs,
                                                                                    entropy_weight=entropy_weight,
                                                                                    kl_weight=kl_weight,
                                                                                    use_FDEADE_aux_loss=True)

                self.optimiser.zero_grad()
                (nll_loss + ade_fde_loss + kl_loss).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimiser.step()

                if i % 100 == 0:
                    self.logger.info(
                        f"Epoch: {epoch:4}, Batch: {i:4} Loss: {ade_fde_loss:.6f} Time: {(time.time() - t_before): 4.4f}")
                    t_before = time.time()

                    eval_loss, fde_loss = self.eval(self.val_loader)

                    if eval_loss < best_eval and fde_loss < best_eval_fde:
                        best_eval = eval_loss
                        best_eval_fde = fde_loss
                        did_epoch_better = True
                        self.logger.info(f"Saving Model with loss:{eval_loss:.4f},{fde_loss:.4f}")
                        torch.save(self.model.state_dict(), self.base_path + f"/model_{epoch}.pth")
                    self.model.train()

            if did_epoch_better:
                self.logger.info(f"Epoch {epoch} was better than last best epoch({last_best_epoch})")
                last_best_epoch = epoch
            if epoch - last_best_epoch > 10:
                self.logger.info(f"Stopping training, no improvement in 10 epochs saved{last_best_epoch}")
                break
            self.optimiser_scheduler.step()

    def eval(self, dataloader):
        eval_loss = 0
        fde_loss = 0
        self.model.eval()
        with torch.no_grad():
            for j, (x_val, y_val) in enumerate(dataloader):
                print(f'\rBatch {j}/{len(dataloader)}', end='')
                ego_in, agents_in, map_lanes, ego_out = self.transform(x_val, y_val)

                pred_obs, mode_probs = self.model(ego_in, agents_in, map_lanes)

                ade_losses, fde_losses, a, f = self._compute_ego_errors(pred_obs, ego_out)

                eval_loss += a / n_pred
                fde_loss += f
        self.model.train()
        eval_loss /= len(dataloader) * batch_size
        fde_loss /= len(dataloader) * batch_size
        return eval_loss, fde_loss


if __name__ == '__main__':
    if "--test" in sys.argv:


        abw_eval = AutoBotWrapperCog(path='./ped_path_predictor/saved_models/60_80/autobots_cognitive.pth')

        dl = singleDatasets(("./ped_path_predictor/data/new_car/01_int_cleaned.npy", "./ped_path_predictor/data/new_car/01_int_cleaned_car.npy"), n_obs, n_pred, 512)
        int_1_a, int_1_f = abw_eval.eval(dl)
        print(f"\nINT-1 {int_1_a:.4f} {int_1_f:.4f}")

        dl = singleDatasets(("./ped_path_predictor/data/new_car/02_int_cleaned.npy", "./ped_path_predictor/data/new_car/02_int_cleaned_car.npy"), n_obs, n_pred, 512)
        int_2_a, int_2_f = abw_eval.eval(dl)
        print(f"\nINT-2 {int_2_a:.4f} {int_2_f:.4f}")

        dl = singleDatasets(("./ped_path_predictor/data/new_car/03_int_cleaned.npy", "./ped_path_predictor/data/new_car/03_int_cleaned_car.npy"), n_obs, n_pred, 512)
        int_3_a, int_3_f = abw_eval.eval(dl)
        print(f"\nINT-3 {int_3_a:.4f} {int_3_f:.4f}")

        dl = singleDatasets(("./ped_path_predictor/data/new_car/04_int_cleaned.npy", "./ped_path_predictor/data/new_car/04_int_cleaned_car.npy"), n_obs, n_pred, 512)
        int_4_a, int_4_f = abw_eval.eval(dl)
        print(f"\nINT-1 {int_4_a:.4f} {int_4_f:.4f}")

        dl = singleDatasets(("./ped_path_predictor/data/new_car/05_int_cleaned.npy", "./ped_path_predictor/data/new_car/05_int_cleaned_car.npy"), n_obs, n_pred, 512)
        int_5_a, int_5_f = abw_eval.eval(dl)
        print(f"\nINT-5 {int_5_a:.4f} {int_5_f:.4f}")

        dl = singleDatasets(("./ped_path_predictor/data/new_car/06_int_cleaned.npy", "./ped_path_predictor/data/new_car/06_int_cleaned_car.npy"), n_obs, n_pred, 512)
        int_6_a, int_6_f = abw_eval.eval(dl)
        print(f"\nINT-6 {int_6_a:.4f} {int_6_f:.4f}")

        # print("\npure test", abw_eval.eval(abw_eval.test_loader))

        dl = singleDatasets(("./ped_path_predictor/data/new_car/01_non_int_cleaned.npy", "./ped_path_predictor/data/new_car/01_non_int_cleaned_car.npy"), n_obs, n_pred, 512)
        non_int_1_a, non_int_1_f = abw_eval.eval(dl)
        print(f"\nNON-INT-1 {non_int_1_a:.4f} {non_int_1_f:.4f}")

        dl = singleDatasets(("./ped_path_predictor/data/new_car/02_non_int_cleaned.npy", "./ped_path_predictor/data/new_car/02_non_int_cleaned_car.npy"), n_obs, n_pred, 512)
        non_int_2_a, non_int_2_f = abw_eval.eval(dl)
        print(f"\nNON-INT-2 {non_int_2_a:.4f} {non_int_2_f:.4f}")

        dl = singleDatasets(("./ped_path_predictor/data/new_car/03_non_int_cleaned.npy", "./ped_path_predictor/data/new_car/03_non_int_cleaned_car.npy"), n_obs,n_pred, 512)
        non_int_3_a, non_int_3_f = abw_eval.eval(dl)
        print(f"\nNON-INT-3 {non_int_3_a:.4f} {non_int_3_f:.4f}")

        dl = singleDatasets(("./ped_path_predictor/data/new_car/04_non_int_cleaned.npy", "./ped_path_predictor/data/new_car/04_non_int_cleaned_car.npy"), n_obs,n_pred, 512)
        non_int_4_a, non_int_4_f = abw_eval.eval(dl)
        print(f"\nNON-INT-4 {non_int_4_a:.4f} {non_int_4_f:.4f}")

        dl = singleDatasets(("./ped_path_predictor/data/new_car/05_non_int_cleaned.npy", "./ped_path_predictor/data/new_car/05_non_int_cleaned_car.npy"), n_obs,n_pred, 512)
        non_int_5_a, non_int_5_f = abw_eval.eval(dl)
        print(f"\nNON-INT-5 {non_int_5_a:.4f} {non_int_5_f:.4f}")

        dl = singleDatasets(("./ped_path_predictor/data/new_car/06_non_int_cleaned.npy", "./ped_path_predictor/data/new_car/06_non_int_cleaned_car.npy"), n_obs,n_pred, 512)
        non_int_6_a, non_int_6_f = abw_eval.eval(dl)
        print(f"\nNON-INT-6 {non_int_6_a:.4f} {non_int_6_f:.4f}")

        print("\n\n\n")
        print("INT")
        print(f"&AutoBot(+Cognitive) & {int_1_a:.4f}/{int_1_f:.4f} & {int_2_a:.4f}/{int_2_f:.4f} & {int_3_a:.4f}/{int_3_f:.4f} & {int_4_a:.4f}/{int_4_f:.4f} & {int_5_a:.4f}/{int_5_f:.4f} & {int_6_a:.4f}/{int_6_f:.4f} \\\\")
        print("NON-INT")
        print(f"&AutoBot(+Cognitive) & {non_int_1_a:.4f}/{non_int_1_f:.4f} & {non_int_2_a:.4f}/{non_int_2_f:.4f} & {non_int_3_a:.4f}/{non_int_3_f:.4f} & {non_int_4_a:.4f}/{non_int_4_f:.4f} & {non_int_5_a:.4f}/{non_int_5_f:.4f} & {non_int_6_a:.4f}/{non_int_6_f:.4f} \\\\")

    else:
        abw = AutoBotWrapperCog()
        abw.train()

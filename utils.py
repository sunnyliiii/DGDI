import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import pywt

def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )
def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))
def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):

    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)
def calc_quantile_CRPS_sum(target, forecast, eval_points, mean_scaler, scaler):

    eval_points = eval_points.mean(-1)
    target = target * scaler + mean_scaler
    target = target.sum(-1)
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = torch.quantile(forecast.sum(-1),quantiles[i],dim=1)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


# Wavelet interpolation
def wavelet_interpolation(source_data, target_data, wavelet='db1', level=3, interp_ratio=0.90, mode='smooth'):
    source_data = source_data.cpu().detach().numpy()
    target_data = target_data.cpu().detach().numpy()
    coeffs_src = pywt.wavedec2(source_data, wavelet, level=level, mode=mode)
    coeffs_tgt = pywt.wavedec2(target_data, wavelet, level=level, mode=mode)

    coeffs_interp = []
    for c_src, c_tgt in zip(coeffs_src, coeffs_tgt):
        if isinstance(c_src, tuple):
            c_interp = tuple(interp_ratio * c_s + (1 - interp_ratio) * c_t for c_s, c_t in zip(c_src, c_tgt))
        else:
            c_interp = interp_ratio * c_src + (1 - interp_ratio) * c_tgt
        coeffs_interp.append(c_interp)

    interpolated_data = pywt.waverec2(coeffs_interp, wavelet, mode=mode)
    expected_shape = (source_data.shape[0], source_data.shape[1], source_data.shape[2])
    if interpolated_data.shape != expected_shape:
        interpolated_data = interpolated_data[:, :, :expected_shape[2]]

    interpolated_data = np.clip(interpolated_data, np.min([source_data, target_data]), np.max([source_data, target_data]))
    return interpolated_data

def train_da(
    src_model,
    tgt_model,
    config,
    weight_decay,
    src_train_loader,
    tgt_train_loader,
    tgt_valid_loader=None,
    src_valid_loader=None,
    valid_epoch_interval=20,
    foldername="",
    miu_ntc=0,
    waveblend_interpolation=1,
    device=None,
):

    src_optimizer = Adam(src_model.parameters(), lr=config["lr"], weight_decay=weight_decay)
    tgt_optimizer = Adam(tgt_model.parameters(), lr=config["lr"], weight_decay=weight_decay)
    if foldername != "":
        tgt_output_path = foldername + "/tgt_model.pth"

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    src_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        src_optimizer, milestones=[p1, p2], gamma=0.1
    )

    tgt_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        tgt_optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        total_loss = 0
        src_model.train()
        tgt_model.train()
        tgt_batch_no = 0

        with tqdm(src_train_loader, mininterval=5.0, maxinterval=50.0) as src_it, \
            tqdm(tgt_train_loader, mininterval=5.0, maxinterval=50.0) as tgt_it:
            for src_batch, tgt_batch in zip(src_it, tgt_it):
                if waveblend_interpolation==1:
                    tgt_observed_data = tgt_batch["observed_data"].to(device).float()
                    src_observed_data = src_batch["observed_data"].to(device).float()
                    local_in_trg = wavelet_interpolation(src_observed_data, tgt_observed_data)
                    tgt_batch["coeff_frequency"] = torch.tensor(local_in_trg).to(device).float()
                    src_batch["coeff_frequency"] = torch.zeros_like(tgt_batch["coeff_frequency"], device=device)
                else: 
                    src_batch["coeff_frequency"] = torch.zeros_like(tgt_batch["observed_data"], device=device)
                    tgt_batch["coeff_frequency"] = torch.zeros_like(tgt_batch["observed_data"], device=device)


                    

                src_optimizer.zero_grad()
                tgt_optimizer.zero_grad()

                src_mse_loss, src_pred, src_feat, src_masks = src_model(src_batch) 
                tgt_mse_loss, tgt_pred, tgt_feat, tgt_masks = tgt_model(tgt_batch)

                total_loss = 1 * src_mse_loss + 1 * tgt_mse_loss


                if miu_ntc != 0: 
                    with torch.no_grad():
                        augmented_tgt_mse_loss, augmented_tgt_pred, augmented_tgt_feat, augmented_tgt_masks = src_model(tgt_batch)                   
                        diff_loss = tgt_mse_loss - augmented_tgt_mse_loss
                        if diff_loss >= 0:
                            diff_loss = diff_loss
                        else:
                            diff_loss = -diff_loss ** 2

                        total_loss += miu_ntc * diff_loss

                total_loss.backward()
                avg_loss += total_loss.item()
                src_optimizer.step()
                tgt_optimizer.step()

                tgt_batch_no += 1

                tgt_it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / tgt_batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
                if tgt_batch_no >= config["itr_per_epoch"]:
                    break

            src_lr_scheduler.step()
            tgt_lr_scheduler.step()

        if tgt_valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            tgt_model.eval()
            avg_loss_valid = 0
            tgt_valid_batch_no = 0

            with torch.no_grad():   
                with tqdm(src_valid_loader, mininterval=5.0, maxinterval=50.0) as valid_src_it, \
                    tqdm(tgt_valid_loader, mininterval=5.0, maxinterval=50.0) as valid_tgt_it:
                    for valid_src_batch, valid_tgt_batch in zip(valid_src_it, valid_tgt_it):
                        if waveblend_interpolation==1:
                            valid_tgt_observed_data = valid_tgt_batch["observed_data"].to(device).float()
                            valid_src_observed_data = valid_src_batch["observed_data"].to(device).float()
    
                            local_in_trg = wavelet_interpolation(valid_src_observed_data, valid_tgt_observed_data)
                            valid_tgt_batch["coeff_frequency"] = torch.tensor(local_in_trg).to(device).float()
                         
                        else: 
                            valid_tgt_batch["coeff_frequency"] = torch.zeros_like(valid_tgt_batch["observed_data"], device=device)
               

                        loss, _, _, _ = tgt_model(valid_tgt_batch, is_train=0)
                        avg_loss_valid += loss.item()
                        tgt_valid_batch_no+=1
                        valid_tgt_it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / tgt_valid_batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / tgt_valid_batch_no,
                    "at",
                    epoch_no,
                )

    if foldername != "":
        torch.save(tgt_model.state_dict(), tgt_output_path)


def evaluate_da(model, test_loader, src_test_loader, adj_src, adj_tgt, nsample=100, scaler=1, mean_scaler=0, foldername="", 
                waveblend_interpolation=1, device=None):

    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []

        tgt_test_batch_no = 0
        with tqdm(src_test_loader, mininterval=5.0, maxinterval=50.0) as src_test_it, \
            tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as tgt_test_it:
            for src_test_batch, tgt_test_batch in zip(src_test_it, tgt_test_it):
                if waveblend_interpolation==1:
                    tgt_observed_data = tgt_test_batch["observed_data"].to(device).float()
                    src_observed_data = src_test_batch["observed_data"].to(device).float()

                
                    local_in_trg = wavelet_interpolation(src_observed_data, tgt_observed_data)
                    tgt_test_batch["coeff_frequency"] = torch.tensor(local_in_trg).to(device).float()
        
                else: 
                    tgt_test_batch["coeff_frequency"] = torch.zeros_like(tgt_test_batch["observed_data"], device=device)

                output = model.evaluate(tgt_test_batch, nsample)


                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points) 
                ) * scaler

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                tgt_test_batch_no += 1 

                tgt_test_it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": tgt_test_batch_no,
                    },
                    refresh=True,
                )

            with open(
                foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                all_target = torch.cat(all_target, dim=0)
                all_evalpoint = torch.cat(all_evalpoint, dim=0)
                all_observed_point = torch.cat(all_observed_point, dim=0)
                all_observed_time = torch.cat(all_observed_time, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )

            CRPS = calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )


            with open(
                foldername + "/target_model_result_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        np.sqrt(mse_total / evalpoints_total),
                        mae_total / evalpoints_total,
                        CRPS,
                    ],
                    f,
                )
                print("target_model_RMSE:", np.sqrt(mse_total / evalpoints_total))
                print("target_model_MAE:", mae_total / evalpoints_total)
                print("target_model_CRPS:", CRPS)
          


                
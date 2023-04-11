import pandas as pd
data = pd.read_csv('image_metrics.csv')

mean_psnr = data['PSNR'].mean()
print(f'Average PSNR : {mean_psnr}')

mean_ssim = data['SSIM'].mean()
print(f'Average SSIM : {mean_ssim}')

mean_msssim = data['MS-SSIM'].mean()
print(f'Average MS-SSIM : {mean_msssim}')

mean_fsim = data['F_SIM'].mean()
print(f'Average F-SIM : {mean_fsim}')

mean_vif = data['VIF'].mean()
print(f'Average VIF : {mean_vif}')

mean_fid = data['FID'].mean()
print(f'Average FID : {mean_fid}')

# Average PSNR : 22.648135892460854
# Average SSIM : 0.8316014530172062
# Average MS-SSIM : 0.9394544590724988
# Average F-SIM : 0.9006765818551239
# Average VIF : 0.5157053168123581
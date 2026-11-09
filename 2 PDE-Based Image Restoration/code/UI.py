import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.fftpack import fft2, ifft2
import gradio as gr
import mysubroutines as sub

def ui_runner(name, method, T, dt, noise_level, blur, K, scheme, L_type):
    plt.close('all')

    # 读取原图
    image_path = f"image/{name}.png"
    u_true = mpimg.imread(image_path)
    if u_true.ndim == 3:
        u_true = np.dot(u_true[..., :3], [0.2989, 0.5870, 0.1140])
    if u_true.max() > 1.0:
        u_true = u_true / 255.0

    # 构造退化图像 f
    np.random.seed(42)
    f = u_true.copy()
    if blur:
        kernel = sub.fspecial([15, 15], 1.0)
        otf_K = sub.psf2otf(kernel, u_true.shape)
        f = np.real(ifft2(fft2(u_true) * otf_K))

    if noise_level > 0:
        f = f + noise_level * np.random.randn(*u_true.shape)
    f = np.clip(f, 0, 1)

    u, max_psnr, max_ssim, best_t, output_image, title_u = None, -1.0, -1.0, 0.0, "", ""

    # 求解 PDE 方程
    if method == "heat":
        output_image = f"image/{name}_heat_T{T}_noise{noise_level}.png"
        u, max_psnr, max_ssim, best_t, _ = sub.solve_heat(f, T=T, dt=dt, u_true=u_true)
        title_u = f"Heat (T={T})"

    elif method == "pm":
        output_image = f"image/{name}_pm_{scheme}_T{T}_noise{noise_level}.png"
        u, max_psnr, max_ssim, best_t, _ = sub.solve_pm(f, T=T, dt=dt, K=K, scheme=scheme, u_true=u_true)
        title_u = f"PM-{scheme} (T={T})"

    elif method == "shock":
        tag = "blur_noise" if noise_level > 0 else "blur_clean"
        output_image = f"image/{name}_shock_{L_type}_T{T}.png"
        u, max_psnr, max_ssim, best_t = sub.solve_shock(f, T=T, dt=dt, L_type=L_type, u_true=u_true)
        title_u = f"Shock-{L_type} (T={T})"

    # 获取图像与日志文本
    fig, log_text = sub.save_comparison(u_true, f, u, max_psnr, max_ssim, best_t, output_image, title_u=title_u)
    
    # 对应终端命令行
    blur_flag = " --blur" if blur else ""
    cmd_str = f"python main.py --name {name} --method {method} --T {T} --dt {dt} --noise_level {noise_level}{blur_flag}"
    if method == "pm":
        cmd_str += f" --K {K} --scheme {scheme}"
    elif method == "shock":
        cmd_str += f" --L_type {L_type}"

    full_output = f"{log_text}\n\n终端命令:\n{cmd_str}"
    return full_output, fig

def toggle_panels(method_val):
    if method_val == "heat":
        return gr.update(visible=False), gr.update(visible=False)
    elif method_val == "pm":
        return gr.update(visible=True), gr.update(visible=False)
    elif method_val == "shock":
        return gr.update(visible=False), gr.update(visible=True)

with gr.Blocks(title="HW2 PDE 调参面板") as demo:
    os.environ['no_proxy'] = 'localhost, 127.0.0.1, ::1'
    os.environ['NO_PROXY'] = 'localhost, 127.0.0.1, ::1'
    gr.Markdown("## HW2: PDE 图像恢复")
    with gr.Row():
        with gr.Column(scale=1):
            name = gr.Radio(["barbara", "peppers256"], value="barbara", label="图片名称 (--name)")
            method = gr.Radio(["heat", "pm", "shock"], value="heat", label="PDE 方程 (--method)")
            
            T = gr.Slider(0.01, 2.0, value=0.2, step=0.01, label="停机时间 (--T)")
            dt = gr.Slider(0.01, 0.2, value=0.01, step=0.01, label="时间步长 (--dt)")
            noise_level = gr.Slider(0.0, 0.05, value=0.02, step=0.01, label="噪声标准差 (--noise_level)")
            blur = gr.Checkbox(value=False, label="开启高斯模糊 (--blur)")
            
            with gr.Group(visible=False) as pm_group:
                gr.Markdown(" Perona-Malik 参数")
                K = gr.Slider(0.01, 0.15, value=0.04, step=0.01, label="梯度阈值 (--K)")
                scheme = gr.Radio(["symmetric", "asymmetric"], value="symmetric", label="离散格式 (--scheme)")
                
            with gr.Group(visible=False) as shock_group:
                gr.Markdown(" Shock Filter 参数")
                L_type = gr.Radio(["u_nn", "laplacian"], value="u_nn", label="算子 L 类型 (--L_type)")

        with gr.Column(scale=2):
            run_btn = gr.Button("Run", variant="primary")
            out_log = gr.Textbox(label="指标结果与对应终端命令", lines=6)
            out_fig = gr.Plot(label="三联对比图")

    method.change(
        fn=toggle_panels,
        inputs=[method],
        outputs=[pm_group, shock_group]
    )

    run_btn.click(
        fn=ui_runner,
        inputs=[name, method, T, dt, noise_level, blur, K, scheme, L_type],
        outputs=[out_log, out_fig]
    )

if __name__ == "__main__":
    demo.launch(server_port=8088, inbrowser=True)

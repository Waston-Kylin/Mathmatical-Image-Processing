import io
import os
import importlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from scipy.fftpack import fft2, ifft2
import gradio as gr
import mysubroutines as sub

def ui_runner(name, method, noise_level, lambda_weight, frame, Level, thresh_type, max_iter, mu, delta, kappa, use_apg):
    importlib.reload(sub)
    plt.close('all')

    if name == "aircraft":
        image_path = f"image/{name}.jpg"
    else:
        image_path = f"image/{name}.png"
    u_true = mpimg.imread(image_path)
    if u_true.ndim == 3:
        u_true = np.dot(u_true[..., :3], [0.2989, 0.5870, 0.1140])
    if u_true.max() > 1.0:
        u_true = u_true / 255.0
    u_true = u_true.astype(np.float64)

    # 退化图像 f (与 HW1 严格对齐: seed=42, 15x15, sigma=1.0)
    np.random.seed(42)
    kernel = sub.fspecial([15, 15], 1.0)
    otf_K = sub.psf2otf(kernel, u_true.shape)
    f_blur = np.real(ifft2(fft2(u_true) * otf_K))
    f = np.clip(f_blur + noise_level * np.random.randn(*u_true.shape), 0.0, 1.0)

    frame_val, level_val, max_it = int(frame), int(Level), int(max_iter)

    if method == "analysis":
        u, max_p, max_s, best_k, history = sub.solve_analysis_admm(
            f, kernel, lambda_weight=lambda_weight, mu=mu, delta=delta,
            frame=frame_val, Level=level_val, max_iter=max_it,
            thresh_type=thresh_type, u_true=u_true
        )
        title_u = f"Analysis-ADMM (F={frame_val}, L={level_val})"
        cmd_extra = f"--frame {frame_val} --Level {level_val} --mu {mu} --delta {delta} --thresh_type {thresh_type}"

    elif method == "balanced":
        u, max_p, max_s, best_k, history = sub.solve_balanced_pfbs(
            f, kernel, lambda_weight=lambda_weight, kappa=kappa,
            frame=frame_val, Level=level_val, max_iter=max_it,
            thresh_type=thresh_type, use_apg=use_apg, u_true=u_true
        )
        alg_tag = "Balanced-APG" if use_apg else "Balanced-PFBS"
        title_u = f"{alg_tag} (F={frame_val}, k={kappa})"
        apg_flag = " --use_apg" if use_apg else ""
        cmd_extra = f"--frame {frame_val} --Level {level_val} --kappa {kappa} --thresh_type {thresh_type}{apg_flag}"

    else:  # tv
        u, max_p, max_s, best_k, history = sub.solve_tv(
            f, kernel, lambda_weight=lambda_weight, mu=mu, max_iter=max_it, u_true=u_true
        )
        title_u = "TV (HW1 Baseline)"
        cmd_extra = f"--mu {mu}"

    if method == "tv" and name == "aircraft":
        output_image = f"../doc/image/{name}_{method}_n{noise_level}.jpg"
    elif method == "tv":
        output_image = f"../doc/image/{name}_{method}_n{noise_level}.png"
    elif method != "tv" and name == "aircraft":
        output_image = f"../doc/image/{name}_{method}_n{noise_level}_f{frame}_t{thresh_type}.jpg"
    else:
        output_image = f"../doc/image/{name}_{method}_n{noise_level}_f{frame}_t{thresh_type}.png"
    fig, log_text = sub.save_comparison(u_true, f, u, max_p, max_s, best_k, output_image=output_image, title_u=title_u)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    rendered_img = Image.open(buf)

    cmd_str = (
        f"python main.py --image_path {image_path} --method {method} "
        f"--noise_level {noise_level} --lambda_weight {lambda_weight} "
        f"--max_iter {max_it} {cmd_extra} --output_image {output_image}"
    )
    return f"{log_text}\n\n终端命令:\n{cmd_str}", rendered_img

def toggle_panels(method_val):
    if method_val == "analysis":
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
    elif method_val == "balanced":
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)
    else:  # tv
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

with gr.Blocks(title="HW4 小波框架图像恢复调参面板") as demo:
    os.environ['no_proxy'] = 'localhost, 127.0.0.1, ::1'
    os.environ['NO_PROXY'] = 'localhost, 127.0.0.1, ::1'
    gr.Markdown("## HW4: 小波紧框架图像去噪去模糊 (Analysis ADMM vs Balanced PFBS vs TV)")

    with gr.Row():
        with gr.Column(scale=1):
            name = gr.Radio(["aircraft", "peppers256", "barbara"], value="aircraft", label="测试图像")
            method = gr.Radio(["analysis", "balanced", "tv"], value="analysis", label="求解模型 (--method)")
            noise_level = gr.Radio([0.01, 0.02], value=0.01, label="高斯噪声标准差 (--noise_level)")

            lambda_weight = gr.Slider(0.0001, 0.01, value=0.0008, step=0.0001, label="正则项权重 (--lambda_weight)")
            max_iter = gr.Slider(1, 150, value=50, step=1, label="最大迭代步数 (--max_iter)")

            with gr.Group(visible=True) as wavelet_group:
                gr.Markdown("**小波紧框架参数**")
                frame = gr.Radio([0, 1, 3], value=1, label="小波阶数 (--frame: 0=Haar, 1=Linear, 3=Cubic)")
                Level = gr.Slider(1, 4, value=2, step=1, label="分解层数 (--Level)")
                thresh_type = gr.Radio(["s", "vs"], value="s", label="阈值算子 (--thresh_type: s=各向异性, vs=各向同性)")

        with gr.Column(scale=2):
            with gr.Group(visible=True) as admm_group:
                gr.Markdown("**Analysis ADMM / TV 参数**")
                mu = gr.Slider(0.002, 0.1, value=0.015, step=0.001, label="惩罚参数 (--mu)")
                delta = gr.Slider(0.1, 1.0, value=1.0, step=0.1, label="对偶步长 (--delta)")

            with gr.Group(visible=False) as pfbs_group:
                gr.Markdown("**Balanced PFBS 参数**")
                kappa = gr.Slider(0.1, 5.0, value=1.0, step=0.1, label="一致性权重 (--kappa)")
                use_apg = gr.Checkbox(value=True, label="开启 Nesterov 加速 (APG / FISTA)")
            run_btn = gr.Button("Run", variant="primary")
            out_log = gr.Textbox(label="指标结果 (Origin / End / Max) 与对应终端命令", lines=6)
            out_fig = gr.Image(label="三联恢复对比图", type="pil", interactive=False)

    method.change(fn=toggle_panels, inputs=[method], outputs=[wavelet_group, admm_group, pfbs_group])
    run_btn.click(
        fn=ui_runner,
        inputs=[name, method, noise_level, lambda_weight, frame, Level, thresh_type, max_iter, mu, delta, kappa, use_apg],
        outputs=[out_log, out_fig]
    )

if __name__ == "__main__":
    demo.launch(server_port=8090, inbrowser=True)
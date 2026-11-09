import io
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from scipy.fftpack import fft2, ifft2
import gradio as gr
import mysubroutines as sub

def get_available_images():
    imgs = ["shapes", "weak_edge"]
    if os.path.exists("image"):
        for f in sorted(os.listdir("image")):
            if f.endswith(".png") and not f.startswith("hw"):
                name = os.path.splitext(f)[0]
                if name not in imgs:
                    imgs.append(name)
    return imgs

def ui_runner(name, method, init_type, max_iter, dt, reinit_int, noise_level, blur, blur_sigma, alpha, edge_k, mu, lam_admm):
    plt.close('all')

    image_path = f"image/{name}.png"
    u_clean = mpimg.imread(image_path)
    if u_clean.ndim == 3:
        u_clean = np.dot(u_clean[..., :3], [0.2989, 0.5870, 0.1140])
    if u_clean.max() > 1.0:
        u_clean = u_clean / 255.0
    u_clean = u_clean.astype(np.float64)

    # 添加模糊 / 噪声
    np.random.seed(42)
    f = u_clean.copy()
    if blur:
        kernel = sub.fspecial([15, 15], blur_sigma)
        otf_K = sub.psf2otf(kernel, u_clean.shape)
        f = np.real(ifft2(fft2(u_clean) * otf_K))

    if noise_level > 0:
        f = f + noise_level * np.random.randn(*u_clean.shape)
    f = np.clip(f, 0.0, 1.0)

    # 对应水平集分割模型
    phi0 = sub.init_level_set(f.shape, init_type=init_type)
    phi, seg_mask, title_m = None, None, ""

    if method == "gac":
        phi, seg_mask = sub.solve_gac(
            f, phi0=phi0, max_iter=int(max_iter), dt=dt,
            alpha=alpha, edge_k=edge_k, reinit_int=int(reinit_int)
        )
        title_m = f"GAC (iter={int(max_iter)})"

    elif method == "cv":
        phi, seg_mask = sub.solve_cv(
            f, phi0=phi0, max_iter=int(max_iter), dt=dt,
            mu=mu, reinit_int=int(reinit_int)
        )
        title_m = f"Original CV (mu={mu})"

    elif method == "convex_cv":
        phi, seg_mask = sub.solve_convex_cv(
            f, max_iter=int(max_iter), mu=mu, lam_admm=lam_admm
        )
        title_m = f"Convex CV (mu={mu})"

    if blur:
        output_image = f"../doc/result/{name}_{method}_{init_type}_{noise_level}_{blur}_{blur_sigma}.png"
    else:
        output_image = f"../doc/result/{name}_{method}_{init_type}_{noise_level}_{blur}.png"
    fig, log_text = sub.save_segmentation_result(u_clean, f, phi, seg_mask, output_image, title_model=title_m)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    rendered_img = Image.open(buf)

    blur_flag = f" --blur --blur_sigma {blur_sigma}" if blur else ""
    cmd_str = f"python main.py --name {name} --method {method} --init_type {init_type} --max_iter {int(max_iter)} --noise_level {noise_level}{blur_flag}"
    if method == "gac":
        cmd_str += f" --dt {dt} --alpha {alpha} --edge_k {edge_k} --reinit_int {int(reinit_int)}"
    elif method == "cv":
        cmd_str += f" --dt {dt} --mu {mu} --reinit_int {int(reinit_int)}"
    elif method == "convex_cv":
        cmd_str += f" --mu {mu} --lam_admm {lam_admm}"

    return f"{log_text}\n\n终端命令:\n{cmd_str}", rendered_img

def toggle_panels(method_val):
    if method_val == "gac":
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)
    elif method_val == "cv":
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)
    elif method_val == "convex_cv":
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

with gr.Blocks(title="HW3 图像分割调参面板") as demo:
    os.environ['no_proxy'] = 'localhost, 127.0.0.1, ::1'
    os.environ['NO_PROXY'] = 'localhost, 127.0.0.1, ::1'
    gr.Markdown("## HW3: 水平集与变分图像分割 (GAC / Chan-Vese / Convexified CV)")

    with gr.Row():
        with gr.Column(scale=1):
            name = gr.Dropdown(choices=get_available_images(), value="shapes", label="测试图像 (--name)")
            method = gr.Radio(["gac", "cv", "convex_cv"], value="gac", label="分割模型 (--method)")
            init_type = gr.Radio(["box", "circle", "corner_circle", "multi_circles"], value="box", label="初始轮廓 (--init_type)")

            max_iter = gr.Slider(500, 5000, value=1000, step=100, label="最大迭代步数 (--max_iter)")
            noise_level = gr.Slider(0.0, 0.10, value=0.0, step=0.01, label="高斯噪声标准差 (--noise_level)")
            blur = gr.Checkbox(value=False, label="开启高斯模糊 (--blur)")
            blur_sigma = gr.Slider(0.5, 5.0, value=2.0, step=0.5, label="模糊核标准差 (--blur_sigma)")

            # 水平集 PDE 推进参数 (GAC & CV)
            with gr.Group(visible=True) as pde_group:
                dt = gr.Slider(0.05, 1.0, value=0.4, step=0.05, label="时间步长 (--dt)")
                reinit_int = gr.Slider(0, 20, value=5, step=1, label="重初始化间隔步数 (--reinit_int)")

            # GAC
            with gr.Group(visible=True) as gac_group:
                gr.Markdown("**GAC 参数**")
                alpha = gr.Slider(-1.0, 1.0, value=-0.2, step=0.1, label="气球力系数 (--alpha)")
                edge_k = gr.Slider(0.01, 0.2, value=0.05, step=0.01, label="边缘梯度阈值 (--edge_k)")

            # CV / Convex-CV
            with gr.Group(visible=False) as cv_group:
                gr.Markdown("**CV / Convex-CV 参数**")
                mu = gr.Slider(0.1, 5.0, value=1.0, step=0.1, label="边界长度/保真平衡参数 (--mu)")
                lam_admm = gr.Slider(0.1, 5.0, value=1.0, step=0.1, label="ADMM 惩罚参数 (--lam_admm)")

        with gr.Column(scale=2):
            run_btn = gr.Button("Run", variant="primary")
            out_log = gr.Textbox(label="分割统计与对应终端命令", lines=5)
            out_fig = gr.Image(label="分割三联对比图 (原图+轮廓 / 二值掩码 / 分片常数重构)", type="pil", interactive=False)

    method.change(
        fn=toggle_panels,
        inputs=[method],
        outputs=[gac_group, cv_group, pde_group]
    )

    run_btn.click(
        fn=ui_runner,
        inputs=[name, method, init_type, max_iter, dt, reinit_int, noise_level, blur, blur_sigma, alpha, edge_k, mu, lam_admm],
        outputs=[out_log, out_fig]
    )

if __name__ == "__main__":
    demo.launch(server_port=8089, inbrowser=True)
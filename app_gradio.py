import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import argparse

import gradio as gr

from predict import load_predictor, predict_image


def build_demo(net, device_str):
    def _predict(img):
        result = predict_image(img, net, device_str)
        classes = {
            f"🐱 猫 ({result['probabilities'][0]:.1%})": result['probabilities'][0],
            f"🐶 狗 ({result['probabilities'][1]:.1%})": result['probabilities'][1],
        }
        label = f"识别结果：{result['class_cn']}（置信度 {result['confidence']:.2%}）"
        return label, classes

    demo = gr.Interface(
        fn=_predict,
        inputs=gr.Image(type="pil", sources=["webcam", "upload"], label="拍照或上传图片"),
        outputs=[
            gr.Textbox(label="识别结果", lines=1),
            gr.Label(label="各类别概率"),
        ],
        title="🐱🐶 猫狗分类器",
        description="基于 ResNet18 迁移学习。对准猫或狗拍照，或上传图片即可识别。",
        examples=None,
    )
    return demo


def main():
    parser = argparse.ArgumentParser(description="启动 Gradio 前端")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--weights", default="weights/best_resnet18.pth")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    net, device = load_predictor(args.weights, args.model, args.device)
    demo = build_demo(net, str(device))
    print(f"\n🚀 Gradio 已就绪 -> http://127.0.0.1:{args.port}")
    print(f"   手机(同局域网)访问: http://<你的电脑IP>:{args.port}\n")
    demo.launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()

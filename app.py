import argparse
import gradio as gr
from common.utils import (
    matcher_zoo,
    change_estimate_geom,
    run_matching,
    ransac_zoo,
    gen_examples,
    DEFAULT_RANSAC,
)

DESCRIPTION = """
# Image Matching WebUI
This Space demonstrates [Image Matching WebUI](https://github.com/Vincentqyw/image-matching-webui) by vincent qin. Feel free to play with it, or duplicate to run image matching without a queue!

🔎 For more details about supported local features and matchers, please refer to https://github.com/Vincentqyw/image-matching-webui

🚀 All algorithms run on CPU for inference on HF, causing slow speeds and high latency. For faster inference, please download the [source code](https://github.com/Vincentqyw/image-matching-webui) for local deployment or check [openxlab space](https://openxlab.org.cn/apps/detail/Realcat/image-matching-webui) and [direct URL](https://g-app-center-083997-7409-n9elr1.openxlab.space) 

🐛 Your feedback is valuable to me. Please do not hesitate to report any bugs [here](https://github.com/Vincentqyw/image-matching-webui/issues).
"""


def ui_change_imagebox(choice):
    return {"value": None, "sources": choice, "__type__": "update"}


def ui_reset_state(
    image0,
    image1,
    match_threshold,
    extract_max_keypoints,
    keypoint_threshold,
    key,
    # enable_ransac=False,
    ransac_method=DEFAULT_RANSAC,
    ransac_reproj_threshold=8,
    ransac_confidence=0.999,
    ransac_max_iter=10000,
    choice_estimate_geom="Homography",
):
    match_threshold = 0.2
    extract_max_keypoints = 1000
    keypoint_threshold = 0.015
    key = list(matcher_zoo.keys())[0]
    image0 = None
    image1 = None
    # enable_ransac = False
    return (
        image0,
        image1,
        match_threshold,
        extract_max_keypoints,
        keypoint_threshold,
        key,
        ui_change_imagebox("upload"),
        ui_change_imagebox("upload"),
        "upload",
        None,  # keypoints
        None,  # raw matches
        None,  # ransac matches
        {},
        {},
        None,
        {},
        DEFAULT_RANSAC,
        8,
        0.999,
        10000,
        "Homography",
    )


# "footer {visibility: hidden}"
def run(config):
    with gr.Blocks(css="style.css") as app:
        gr.Markdown(DESCRIPTION)

        with gr.Row(equal_height=False):
            with gr.Column():
                with gr.Row():
                    matcher_list = gr.Dropdown(
                        choices=list(matcher_zoo.keys()),
                        value="disk+lightglue",
                        label="Matching Model",
                        interactive=True,
                    )
                    match_image_src = gr.Radio(
                        ["upload", "webcam", "canvas"],
                        label="Image Source",
                        value="upload",
                    )
                with gr.Row():
                    input_image0 = gr.Image(
                        label="Image 0",
                        type="numpy",
                        image_mode="RGB",
                        height=300,
                        interactive=True,
                    )
                    input_image1 = gr.Image(
                        label="Image 1",
                        type="numpy",
                        image_mode="RGB",
                        height=300,
                        interactive=True,
                    )

                with gr.Row():
                    button_reset = gr.Button(value="Reset")
                    button_run = gr.Button(value="Run Match", variant="primary")

                with gr.Accordion("Advanced Setting", open=False):
                    with gr.Accordion("Matching Setting", open=True):
                        with gr.Row():
                            match_setting_threshold = gr.Slider(
                                minimum=0.0,
                                maximum=1,
                                step=0.001,
                                label="Match thres.",
                                value=0.1,
                            )
                            match_setting_max_features = gr.Slider(
                                minimum=10,
                                maximum=10000,
                                step=10,
                                label="Max features",
                                value=1000,
                            )
                        # TODO: add line settings
                        with gr.Row():
                            detect_keypoints_threshold = gr.Slider(
                                minimum=0,
                                maximum=1,
                                step=0.001,
                                label="Keypoint thres.",
                                value=0.015,
                            )
                            detect_line_threshold = gr.Slider(
                                minimum=0.1,
                                maximum=1,
                                step=0.01,
                                label="Line thres.",
                                value=0.2,
                            )
                        # matcher_lists = gr.Radio(
                        #     ["NN-mutual", "Dual-Softmax"],
                        #     label="Matcher mode",
                        #     value="NN-mutual",
                        # )
                    with gr.Accordion("RANSAC Setting", open=True):
                        with gr.Row(equal_height=False):
                            # enable_ransac = gr.Checkbox(label="Enable RANSAC")
                            ransac_method = gr.Dropdown(
                                choices=ransac_zoo.keys(),
                                value=DEFAULT_RANSAC,
                                label="RANSAC Method",
                                interactive=True,
                            )
                        ransac_reproj_threshold = gr.Slider(
                            minimum=0.0,
                            maximum=12,
                            step=0.01,
                            label="Ransac Reproj threshold",
                            value=8.0,
                        )
                        ransac_confidence = gr.Slider(
                            minimum=0.0,
                            maximum=1,
                            step=0.00001,
                            label="Ransac Confidence",
                            value=0.99999,
                        )
                        ransac_max_iter = gr.Slider(
                            minimum=0.0,
                            maximum=100000,
                            step=100,
                            label="Ransac Iterations",
                            value=10000,
                        )

                    with gr.Accordion("Geometry Setting", open=False):
                        with gr.Row(equal_height=False):
                            # show_geom = gr.Checkbox(label="Show Geometry")
                            choice_estimate_geom = gr.Radio(
                                ["Fundamental", "Homography"],
                                label="Reconstruct Geometry",
                                value="Homography",
                            )

                # with gr.Column():
                # collect inputs
                inputs = [
                    input_image0,
                    input_image1,
                    match_setting_threshold,
                    match_setting_max_features,
                    detect_keypoints_threshold,
                    matcher_list,
                    # enable_ransac,
                    ransac_method,
                    ransac_reproj_threshold,
                    ransac_confidence,
                    ransac_max_iter,
                    choice_estimate_geom,
                ]

                # Add some examples
                with gr.Row():
                    # Example inputs
                    gr.Examples(
                        examples=gen_examples(),
                        inputs=inputs,
                        outputs=[],
                        fn=run_matching,
                        cache_examples=False,
                        label=(
                            "Examples (click one of the images below to Run"
                            " Match)"
                        ),
                    )
                with gr.Accordion("Open for More!", open=False):
                    gr.Markdown(
                        f"""
                        <h3>Supported Algorithms</h3>
                        {", ".join(matcher_zoo.keys())}
                        """
                    )

            with gr.Column():
                output_keypoints = gr.Image(label="Keypoints", type="numpy")
                output_matches_raw = gr.Image(label="Raw Matches", type="numpy")
                output_matches_ransac = gr.Image(
                    label="Ransac Matches", type="numpy"
                )
                with gr.Accordion(
                    "Open for More: Matches Statistics", open=False
                ):
                    matches_result_info = gr.JSON(label="Matches Statistics")
                    matcher_info = gr.JSON(label="Match info")

                with gr.Accordion("Open for More: Warped Image", open=False):
                    output_wrapped = gr.Image(
                        label="Wrapped Pair", type="numpy"
                    )
                    with gr.Accordion(
                        "Open for More: Geometry info", open=False
                    ):
                        geometry_result = gr.JSON(
                            label="Reconstructed Geometry"
                        )

            # callbacks
            match_image_src.change(
                fn=ui_change_imagebox,
                inputs=match_image_src,
                outputs=input_image0,
            )
            match_image_src.change(
                fn=ui_change_imagebox,
                inputs=match_image_src,
                outputs=input_image1,
            )

            # collect outputs
            outputs = [
                output_keypoints,
                output_matches_raw,
                output_matches_ransac,
                matches_result_info,
                matcher_info,
                geometry_result,
                output_wrapped,
            ]
            # button callbacks
            button_run.click(fn=run_matching, inputs=inputs, outputs=outputs)

            # Reset images
            reset_outputs = [
                input_image0,
                input_image1,
                match_setting_threshold,
                match_setting_max_features,
                detect_keypoints_threshold,
                matcher_list,
                input_image0,
                input_image1,
                match_image_src,
                output_keypoints,
                output_matches_raw,
                output_matches_ransac,
                matches_result_info,
                matcher_info,
                output_wrapped,
                geometry_result,
                # enable_ransac,
                ransac_method,
                ransac_reproj_threshold,
                ransac_confidence,
                ransac_max_iter,
                choice_estimate_geom,
            ]
            button_reset.click(
                fn=ui_reset_state, inputs=inputs, outputs=reset_outputs
            )

            # estimate geo
            choice_estimate_geom.change(
                fn=change_estimate_geom,
                inputs=[
                    input_image0,
                    input_image1,
                    geometry_result,
                    choice_estimate_geom,
                ],
                outputs=[output_wrapped, geometry_result],
            )

    app.queue().launch(share=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.yaml",
        help="configuration file path",
    )
    args = parser.parse_args()
    config = None
    run(config)

if __name__ == "__main__":
    from train_jittor import DMANetDetection
    import argparse
    import warnings
    from config.settings import Settings

    parser = argparse.ArgumentParser(description="Test network.")
    parser.add_argument("--settings_file", type=str, default="./config/settings.yaml",
                        help="Path to settings yaml")
    parser.add_argument("--weights", type=str,
                        default="/root/code/AAAI_Event_based_detection/log/20250716-212918/checkpoints/model_step_30.pth",
                        help="model.pth path(s)")
    parser.add_argument("--conf_thresh", type=float, default=0.1,
                        help="object confidence threshold")
    parser.add_argument("--iou_thresh", type=float, default=0.5,
                        help="IoU threshold for NMS")
    parser.add_argument("--save_npy", type=bool, default=True,
                        help="save detection results(predicted bounding boxes), .npy file for visualization")

    args = parser.parse_args()

    settings = Settings(args.settings_file, generate_log=True)
    trainer = DMANetDetection(settings)
    trainer.loadCheckpoint("/root/code/DMANet-Jittor/log/20250722-122307/checkpoints/model_step_30.pth")
    trainer.test()
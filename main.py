import argparse
import sys
import os

# Ensure the root directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description="S-JEPA Skeleton Training & Evaluation Center")
    parser.add_argument(
        "--mode", 
        type=str, 
        required=True, 
        choices=["pretrain", "finetune", "preprocess"],
        help="Chọn chế độ: pretrain | finetune | preprocess"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Đường dẫn file config YAML. Ví dụ: configs/pretrain_ntu60_xsub.yaml hoặc configs/finetune_ntu60_xsub.yaml"
    )
    
    args = parser.parse_args()
    
    if args.mode == "pretrain":
        print(">>> KHỞI CHẠY GIAI ĐOẠN PRE-TRAINING (FOUNDATION MODEL)...")
        from train import main as pretrain_main
        pretrain_main(config_path=args.config)
        
    elif args.mode == "finetune":
        print(">>> KHỞI CHẠY GIAI ĐOẠN FINE-TUNING (ACTION RECOGNITION)...")
        from finetune import main_finetune
        main_finetune(config_path=args.config)
        
    elif args.mode == "preprocess":
        print(">>> KHỞI CHẠY GIAI ĐOẠN TIỀN XỬ LÝ DỮ LIỆU (SKELETON -> NPY)...")
        from scripts.preprocess_ntu import main as preprocess_main
        preprocess_main()

if __name__ == "__main__":
    main()

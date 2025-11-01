import argparse
from .pipeline import run_pipeline

def main():
    p = argparse.ArgumentParser("sentimentarcs")
    p.add_argument("input")
    p.add_argument("--model", default="vader")
    p.add_argument("--clean-tech", default=None)
    p.add_argument("--smooth-method", default=None)
    args = p.parse_args()
    run_pipeline(args.input, model_name=args.model, clean_tech=args.clean_tech, smooth_method=args.smooth_method)

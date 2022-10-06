from load import load_dataset
from model import fit_predict
from outline import draw_outline
from preprocess import preprocess


def main():
    preprocess(input_dir="input", input_csv="data.csv", output_dir="output")
    raw_train_ds, raw_val_ds, raw_test_ds = load_dataset("output")
    history = fit_predict(raw_train_ds, raw_val_ds, raw_test_ds)
    draw_outline(history)
    pass


if __name__ == '__main__':
    main()

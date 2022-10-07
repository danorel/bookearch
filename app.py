from load import load_dataset
from model import fit_predict
from analysis import draw_outline, test_model
from pipeline import pipe


def main():
    pipe(input_dir="input", input_csv="data.csv", output_dir="output")
    raw_train_ds, raw_val_ds = load_dataset("output")
    history, export_model = fit_predict(raw_train_ds, raw_val_ds)
    test_model(export_model)
    draw_outline(history)
    pass


if __name__ == '__main__':
    main()

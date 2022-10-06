from model import load_dataset, fit, predict


def main():
    train_input_fn, test_input_fn = load_dataset("dataset")
    # estimator = fit(train_input_fn=train_input_fn)
    # predict(estimator=estimator, test_input_fn=test_input_fn)


if __name__ == '__main__':
    main()

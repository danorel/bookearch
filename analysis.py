import matplotlib.pyplot as plt


def test_model(export_model):
    prediction = export_model.predict(["The chief objectives of the Viking Mission to Mars were to get pictures and samples of soil to find out if there were signs of life there. Tilloo and his parents lived beneath our planet under artificial conditions. They were not allowed to go on the surface for two reasons. The air was too thin to breathe and the temperature was very low. Tilloo’s father was a part of that team. He used to go to work daily through a secret passage. But it was a forbidden route for the boy. Tilloo was curious to know about and see the sky. His mother would not let him go."])
    print(prediction)
    pass


def draw_outline(history):
    history_dict = history.history
    history_dict.keys()

    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    pass

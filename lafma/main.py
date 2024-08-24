from train import train

def main():
    # Set hyperparameters
    num_epochs = 10
    learning_rate = 1e-4
    batch_size = 4

    # Run the training process
    train(num_epochs=num_epochs, learning_rate=learning_rate, batch_size=batch_size)

if __name__ == "__main__":
    main()

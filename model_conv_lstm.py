from project_3_models import ConvLSTM, train, evaluate, train_loader, test_loader

if __name__ == "__main__":
    print("Training ConvLSTM")
    model = ConvLSTM()
    train(model, train_loader)
    print("Evaluating ConvLSTM")
    evaluate(model, test_loader)
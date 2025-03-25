from project_3_models import TCNN, train, evaluate, train_loader, test_loader

if __name__ == "__main__":
    print("Training TCNN")
    model = TCNN()
    train(model, train_loader)
    print("Evaluating TCNN")
    evaluate(model, test_loader)

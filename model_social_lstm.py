from project_3_models import SocialLSTM, train, evaluate, train_loader, test_loader

if __name__ == "__main__":
    print("Training SocialLSTM")
    model = SocialLSTM()
    train(model, train_loader)
    print("Evaluating SocialLSTM")
    evaluate(model, test_loader)

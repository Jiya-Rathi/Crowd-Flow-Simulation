from project_3_models import SocialLSTM, train, evaluate, train_loader, test_loader
import torch

if __name__ == "__main__":
    print("Training SocialLSTM")
    model = SocialLSTM()
    train(model, train_loader)
    torch.save(model.state_dict(), "social_lstm_checkpoint.pth")
    print("Model saved to social_lstm_checkpoint.pth")
    print("Evaluating SocialLSTM")
    evaluate(model, test_loader)

from project_3_models import ConvLSTM, train, evaluate, train_loader, test_loader
import torch

if __name__ == "__main__":
    print("Training ConvLSTM")
    model = ConvLSTM()
    train(model, train_loader)
    torch.save(model.state_dict(), "conv_lstm_checkpoint.pth")
    print("Model saved to conv_lstm_checkpoint.pth")
    print("Evaluating SocialLSTM")
    evaluate(model, test_loader)
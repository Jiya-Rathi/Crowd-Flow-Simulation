from project_3_models import TCNN, train, evaluate, train_loader, test_loader
import torch

if __name__ == "__main__":
    print("Training TCNN")
    model = TCNN()
    train(model, train_loader)
    torch.save(model.state_dict(), "tcnn_checkpoint.pth")
    print("Model saved to tcnn_checkpoint.pth")
    print("Evaluating TCNN")
    evaluate(model, test_loader)

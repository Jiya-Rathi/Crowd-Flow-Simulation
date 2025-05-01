from project_3_models import TrajectoryLSTM, train, evaluate, train_loader, test_loader
import torch

if __name__ == "__main__":
    print("Training TrajectoryLSTM")
    model = TrajectoryLSTM()
    train(model, train_loader)
    torch.save(model.state_dict(), "trajectory_lstm.pth")
    print("Model saved to trajectory_lstm.pth")
    print("Evaluating SocialLSTM")
    evaluate(model, test_loader)
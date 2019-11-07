from train import Train

if __name__ == "__main__":
    app = Train()
    #对训练集进行训练
    app.train()
    #对测试机进行测试
    app.calculate_accuracy()
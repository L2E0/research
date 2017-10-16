import model

def summary():
    model.Generator().summary()
    model.Discriminator().summary()

if __name__ == "__main__":
    summary()
if [ ! -d "cifar-10-batches-py" ]; then
  download() {
    if command -v wget >/dev/null 2>&1; then
      wget "$1" -O "$2"
    else
      curl -L "$1" -o "$2"
    fi
  }

  download http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz cifar-10-python.tar.gz
  tar -xzvf cifar-10-python.tar.gz
  rm cifar-10-python.tar.gz
  download http://cs231n.stanford.edu/imagenet_val_25.npz imagenet_val_25.npz
fi

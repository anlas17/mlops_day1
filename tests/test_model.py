import sys
import torch
import pytest
import os.path
sys.path.append("C:/Users/zebra/.cookiecutters/mlops_day1/src/models/")
sys.path.append("C:/Users/zebra/.cookiecutters/mlops_day1/")
from model import MyAwesomeModel

model = MyAwesomeModel()
tensor_random = torch.rand(1,1,28,28)

@pytest.mark.skipif(not os.path.exists('data/raw/test.npz'), reason='file not found')
def test_error_on_wrong_shape():
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        model(torch.randn(1,2,3))

assert model(tensor_random).shape == (1,10), 'Tensor has a wrong shape'

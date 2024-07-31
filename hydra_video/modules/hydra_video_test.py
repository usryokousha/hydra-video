


import unittest
import torch
from mamba.mamba_ssm.utils.generation import InferenceParams
from hydra_video.modules.hydra_video import HydraVideo, flip_frames, get_seq_idx


class TestHydraVideo(unittest.TestCase):
    def setUp(self):
        self.d_model = 256
        self.d_state = 64
        self.num_frames = 8
        self.seq_len = 32
        self.batch_size = 2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HydraVideo(d_model=self.d_model, d_state=self.d_state, layer_idx=0).to(
            self.device
        )

    def test_hydra_video_forward(self):
        input_tensor = torch.randn(self.batch_size, self.seq_len, self.d_model).to(
            self.device
        )
        output = self.model(input_tensor, num_frames=self.num_frames)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_flip_frames(self):
        x = torch.arange(24).reshape(1, 24, 1)
        flipped = flip_frames(x, num_frames=4)
        print(flipped)
        expected = torch.tensor(
            [
                [
                    [3],
                    [2],
                    [1],
                    [0],
                    [7],
                    [6],
                    [5],
                    [4],
                    [11],
                    [10],
                    [9],
                    [8],
                    [15],
                    [14],
                    [13],
                    [12],
                    [19],
                    [18],
                    [17],
                    [16],
                    [23],
                    [22],
                    [21],
                    [20],
                ]
            ]
        ).long()
        self.assertTrue(torch.allclose(flipped, expected))

    def test_get_seq_idx(self):
        seq_idx = get_seq_idx(batch=2, seq_len=8, num_frames=4)
        expected = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3], [4, 4, 5, 5, 6, 6, 7, 7]])
        self.assertTrue(torch.equal(seq_idx, expected))

    def test_hydra_video_inference(self):
        input_tensor = torch.randn(2, 32, self.d_model).to(self.device)
        inference_params = InferenceParams(max_seqlen=32, max_batch_size=2)
        
        output = self.model(
            input_tensor, num_frames=8, inference_params=inference_params
        )
        self.assertEqual(output.shape, (2, 32, self.d_model))


if __name__ == "__main__":
    unittest.main()

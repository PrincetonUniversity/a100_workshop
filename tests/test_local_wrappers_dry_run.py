import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class LocalWrapperDryRunTests(unittest.TestCase):
    def assert_dry_run(
        self, relative_path: str, expected_path: str, expected_command: str
    ) -> None:
        script_path = REPO_ROOT / relative_path
        self.assertTrue(script_path.exists(), f"Missing script: {relative_path}")

        result = subprocess.run(
            [str(script_path), "--dry-run"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(expected_path, result.stdout)
        self.assertIn(expected_command, result.stdout)

    def test_cuda_wrapper_dry_run(self) -> None:
        self.assert_dry_run(
            "scripts/local/run_03_cuda.sh",
            "03_gpu_programming_review/code",
            "nvcc -O3 -arch=sm_80 -o vector_addition vector_addition.cu",
        )

    def test_openacc_wrapper_dry_run(self) -> None:
        self.assert_dry_run(
            "scripts/local/run_03_openacc.sh",
            "03_gpu_programming_review/code",
            "nvc -acc -gpu=cc80 -Minfo=all -o vector_addition_openacc vector_addition.c",
        )

    def test_cupy_wrapper_dry_run(self) -> None:
        self.assert_dry_run(
            "scripts/local/run_06_cupy.sh",
            "06_cupy/code",
            "python myscript.py",
        )

    def test_pytorch_wrapper_dry_run(self) -> None:
        self.assert_dry_run(
            "scripts/local/run_07_pytorch.sh",
            "07_pytorch/code",
            "python myscript.py",
        )

    def test_tensorflow_wrapper_dry_run(self) -> None:
        self.assert_dry_run(
            "scripts/local/run_08_tensorflow.sh",
            "08_tensorflow/code",
            "python tf_gpu_smoke.py",
        )


if __name__ == "__main__":
    unittest.main()

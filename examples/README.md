# Examples

每个预置放一组 384×384 的样例（`*_000`），方便不装环境也能先看效果：

| 文件 | 内容 |
| --- | --- |
| `<preset>_000.png` | SEM 风格灰度图 |
| `<preset>_000_mask.png` | 真值实例掩膜（调色板 PNG） |
| `<preset>_000_overlay.png` | 掩膜叠加 |

重新生成（含完整 4 张 + CSV + 预览图）：

```bash
python -m micrograph_synth.cli --preset grains --count 4 --size 384 --out examples/grains
python tools/make_showcase.py        # 重新生成 docs/showcase.png
```

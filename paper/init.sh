#!/bin/bash
# 初始化新论文项目（project/paper 结构）
# 用法: bash init.sh /path/to/my-project "Paper Title"
#
# 生成结构:
#   my-project/
#   ├── .claude/skills/   ← skills 在项目根目录
#   └── paper/            ← 论文源文件

set -e

TEMPLATE_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT="${1:?用法: bash init.sh /path/to/my-project \"Paper Title\"}"
TITLE="${2:-YOUR PAPER TITLE}"

PAPER_DIR="$PROJECT/paper"

# 创建项目目录
mkdir -p "$PROJECT"

# 复制模板到 paper/ 子目录
if [ -d "$PAPER_DIR" ]; then
    echo "ERROR: $PAPER_DIR 已存在"
    exit 1
fi
cp -r "$TEMPLATE_DIR" "$PAPER_DIR"
rm -f "$PAPER_DIR/init.sh"

# 将 skills 合并到项目根目录（不覆盖已有文件）
if [ -d "$PROJECT/.claude" ]; then
    cp -rn "$PAPER_DIR/.claude/"* "$PROJECT/.claude/" 2>/dev/null || true
    rm -rf "$PAPER_DIR/.claude"
    echo "  ⚠ 已有 .claude/ 目录，skills 已合并（未覆盖已有文件）"
else
    mv "$PAPER_DIR/.claude" "$PROJECT/.claude"
fi

# 替换标题
sed -i "s/YOUR PAPER TITLE/${TITLE}/" "$PAPER_DIR/main.tex"

SKILL_COUNT=$(ls "$PROJECT/.claude/skills/" 2>/dev/null | wc -l)

echo "✓ 论文项目已创建: $PROJECT"
echo "  模板: NeurIPS 2026"
echo "  标题: $TITLE"
echo "  Skills: $SKILL_COUNT 个 (在 $PROJECT/.claude/skills/)"
echo "  论文: $PAPER_DIR/"
echo ""
echo "下一步:"
echo "  cd $PROJECT"
echo "  cd paper && bash build.sh  # 编译 PDF"

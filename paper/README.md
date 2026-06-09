# Paper Workspace Template

## 用法

```bash
# 1. 在项目目录下，将模板作为 paper/ 子目录
mkdir my-project && cd my-project
cp -r /mnt/dhwfile/raise/user/linhonglin/paper-workspace-template paper

# 2. ⚠️ 重要：将 skills 合并到项目根目录，否则 Claude Code 无法加载
#    如果项目根目录已有 .claude/，用 cp -rn 合并（不覆盖已有文件）
if [ -d .claude ]; then
    cp -rn paper/.claude/* .claude/
    rm -rf paper/.claude
else
    mv paper/.claude .claude
fi

# 3. 修改 paper/main.tex 的标题和内容
# 4. 编译
cd paper && bash build.sh
```

> **为什么要移动 `.claude/` 目录？**
> Claude Code 只加载**项目根目录**下的 `.claude/skills/`。模板作为 `paper/` 子目录使用时，skills 在 `paper/.claude/skills/` 下，不会被自动加载。必须将 `.claude/` 移到项目根目录。

## 项目结构

```
my-project/                   # 项目根目录
├── .claude/skills/           # ← skills 必须在这里
│   ├── paper-figure/
│   ├── paper-plan/
│   ├── paper-write/
│   ├── paper-compile/
│   ├── research-paper-writing/
│   ├── scientific-visualization/
│   ├── scientific-writing/
│   ├── paper-strategist/
│   └── paper-composer/
├── paper/                    # 论文目录 (本模板)
│   ├── main.tex              # 论文主文件 (NeurIPS 2026 格式)
│   ├── build.sh              # 编译脚本，PDF 命名为项目名
│   ├── neurips_2026.sty
│   ├── mybst.bst
│   ├── refs.bib
│   ├── sections/
│   │   ├── 0_abstract.tex
│   │   ├── 1_introduction.tex
│   │   ├── 2_related_work.tex
│   │   ├── 3_method.tex
│   │   ├── 4_experiments.tex
│   │   ├── 5_conclusion.tex
│   │   └── A_appendix.tex
│   ├── figures/
│   └── tables/
├── NARRATIVE_REPORT.md       # 研究叙述 (可选)
├── PAPER_PLAN.md             # 论文大纲 (由 /paper-plan 生成)
└── ...
```

## 固定 5-Section 结构

| § | Section | 页数 |
|---|---------|------|
| 1 | Introduction | 1.5p |
| 2 | Related Work | ≤1p (2-3 categories, citation-dense) |
| 3 | Method | 2.5p |
| 4 | Experiments | 3p |
| 5 | Conclusion | 0.5p |

## 可用 Skills

| Skill | 用途 |
|-------|------|
| `/paper-plan` | 生成论文大纲 (claims-evidence matrix) |
| `/paper-write` | 按章节结构化撰写 LaTeX |
| `/paper-figure` | 生成 publication-ready 图表 |
| `/paper-compile` | 编译 PDF，自动修复错误 |
| `/research-paper-writing` | 段落级润色、self-review |
| `/paper-strategist` | 论文整体规划和 story 设计 |

## 编译

```bash
cd paper && bash build.sh
# 输出: my-project.pdf (以项目目录命名)
```

需要 texlive。如果集群上没有：

```bash
# 用 apptainer
apptainer exec /path/to/texlive.sif bash build.sh
```

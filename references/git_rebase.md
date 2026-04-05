example of merging feature branch "refactor/dev/cli-py" back into dev 

git switch refactor/dev/cli-py
git fetch origin
git rebase origin/dev

gh pr create --base dev --head refactor/dev/cli-py --fill

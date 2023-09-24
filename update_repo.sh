#> updates entire GitHub repository

git add .

echo 'enter commit message:'
read commitMessage

git commit -m "$commitMessage"

branch="main"

git push origin $branch

read

# note: remember to press enter afterwards...
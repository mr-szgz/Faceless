git checkout origin/main examples
# TODO: if ./examples/faceless then remove
Remove-Item -Path ./examples/faceless -Recurse -Force
if (Test-Path -Path ./examples/labels) {
  Remove-Item -Path ./examples/labels -Recurse -Force
}

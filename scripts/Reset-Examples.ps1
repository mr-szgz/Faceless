git checkout origin/main examples
if (Test-Path -Path ./examples/faceless) {
  Remove-Item -Path ./examples/faceless -Recurse -Force
}
if (Test-Path -Path ./examples/labels) {
  Remove-Item -Path ./examples/labels -Recurse -Force
}

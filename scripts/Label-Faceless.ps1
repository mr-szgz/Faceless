function Label-Faceless {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Path,
        [float]$Conf = 0.2,
        [string]$Model = "yolov8n-oiv7.pt"
    )

    & yolo detect predict model=$Model `
        save=False `
        save_txt=True `
        save_conf=True `
        exist_ok=True `
        verbose=True `
        vid_stride=50 `
        conf=$Conf `
        name="." `
        project=$Path `
        source=$Path
}

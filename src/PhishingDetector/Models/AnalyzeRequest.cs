using System.ComponentModel.DataAnnotations;

namespace PhishingDetector.Models;

public class AnalyzeRequest
{
    [Required(ErrorMessage = "La URL es obligatoria.")]
    public string Url { get; set; } = string.Empty;
}

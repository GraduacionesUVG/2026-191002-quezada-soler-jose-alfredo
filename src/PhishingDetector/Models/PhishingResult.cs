using System.Text.Json.Serialization;

namespace PhishingDetector.Models;

public class PhishingResult
{
    [JsonPropertyName("is_phishing")]
    public bool IsPhishing { get; set; }

    [JsonPropertyName("probability")]
    public double Probability { get; set; }

    [JsonPropertyName("risk_level")]
    public string RiskLevel { get; set; } = string.Empty;

    [JsonPropertyName("features")]
    public Dictionary<string, double>? Features { get; set; }
}

public class AnalysisResponse
{
    public int Id { get; set; }
    public string Url { get; set; } = string.Empty;
    public bool IsPhishing { get; set; }
    public double Probability { get; set; }
    public string RiskLevel { get; set; } = string.Empty;
    public DateTime CreatedAt { get; set; }
    public Dictionary<string, double>? Features { get; set; }
}

public class DailyCount
{
    public string Date { get; set; } = string.Empty;
    public int Total { get; set; }
    public int Phishing { get; set; }
}

public class DashboardMetrics
{
    public int TotalAnalyses { get; set; }
    public int PhishingDetected { get; set; }
    public int LegitimateDetected { get; set; }
    public double AverageProbability { get; set; }
    public int HighRisk { get; set; }
    public int MediumRisk { get; set; }
    public int LowRisk { get; set; }
    public List<Analysis> RecentAnalyses { get; set; } = new();
    public List<DailyCount> DailyCounts { get; set; } = new();
}

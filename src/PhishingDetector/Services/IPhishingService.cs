using PhishingDetector.Models;

namespace PhishingDetector.Services;

public interface IPhishingService
{
    Task<AnalysisResponse> AnalyzeUrlAsync(string url);
    Task<List<Analysis>> GetHistoryAsync(int limit = 50);
    Task<DashboardMetrics> GetMetricsAsync();
}

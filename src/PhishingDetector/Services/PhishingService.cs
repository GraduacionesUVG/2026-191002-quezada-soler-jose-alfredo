using Microsoft.EntityFrameworkCore;
using PhishingDetector.Data;
using PhishingDetector.Models;

namespace PhishingDetector.Services;

public class PhishingService : IPhishingService
{
    private readonly AppDbContext _db;
    private readonly FastApiClient _ai;
    private readonly ILogger<PhishingService> _log;

    public PhishingService(AppDbContext db, FastApiClient ai, ILogger<PhishingService> log)
    {
        _db = db;
        _ai = ai;
        _log = log;
    }

    public async Task<AnalysisResponse> AnalyzeUrlAsync(string url)
    {
        var aiResult = await _ai.AnalyzeAsync(url);

        var record = new Analysis
        {
            Url = url,
            Result = aiResult.IsPhishing ? "PHISHING" : "LEGITIMATE",
            Probability = aiResult.Probability,
            RiskLevel = aiResult.RiskLevel,
            CreatedAt = DateTime.UtcNow,
        };

        _db.Analyses.Add(record);
        await _db.SaveChangesAsync();

        return new AnalysisResponse
        {
            Id = record.Id,
            Url = url,
            IsPhishing = aiResult.IsPhishing,
            Probability = aiResult.Probability,
            RiskLevel = aiResult.RiskLevel,
            CreatedAt = record.CreatedAt,
            Features = aiResult.Features,
        };
    }

    public async Task<List<Analysis>> GetHistoryAsync(int limit = 50) =>
        await _db.Analyses
            .OrderByDescending(a => a.CreatedAt)
            .Take(limit)
            .ToListAsync();

    public async Task<DashboardMetrics> GetMetricsAsync()
    {
        var all = await _db.Analyses.ToListAsync();
        var recent = await _db.Analyses
            .OrderByDescending(a => a.CreatedAt)
            .Take(10)
            .ToListAsync();

        var daily = all
            .GroupBy(a => a.CreatedAt.Date)
            .OrderBy(g => g.Key)
            .TakeLast(7)
            .Select(g => new DailyCount
            {
                Date = g.Key.ToString("dd/MM"),
                Total = g.Count(),
                Phishing = g.Count(x => x.Result == "PHISHING"),
            })
            .ToList();

        return new DashboardMetrics
        {
            TotalAnalyses = all.Count,
            PhishingDetected = all.Count(a => a.Result == "PHISHING"),
            LegitimateDetected = all.Count(a => a.Result == "LEGITIMATE"),
            AverageProbability = all.Any() ? all.Average(a => a.Probability) : 0,
            HighRisk = all.Count(a => a.RiskLevel == "HIGH"),
            MediumRisk = all.Count(a => a.RiskLevel == "MEDIUM"),
            LowRisk = all.Count(a => a.RiskLevel == "LOW"),
            RecentAnalyses = recent,
            DailyCounts = daily,
        };
    }
}

using Microsoft.AspNetCore.Mvc;
using PhishingDetector.Services;

namespace PhishingDetector.Controllers;

public class DashboardController : Controller
{
    private readonly IPhishingService _svc;

    public DashboardController(IPhishingService svc) => _svc = svc;

    public async Task<IActionResult> Index()
    {
        var metrics = await _svc.GetMetricsAsync();
        return View(metrics);
    }

    [HttpGet("api/dashboard/metrics")]
    public async Task<IActionResult> Metrics()
    {
        var metrics = await _svc.GetMetricsAsync();
        return Ok(metrics);
    }
}

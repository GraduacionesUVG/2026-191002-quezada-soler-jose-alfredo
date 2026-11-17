using Microsoft.AspNetCore.Mvc;
using PhishingDetector.Models;
using PhishingDetector.Services;

namespace PhishingDetector.Controllers;

[ApiController]
[Route("api/phishing")]
public class PhishingApiController : ControllerBase
{
    private readonly IPhishingService _svc;

    public PhishingApiController(IPhishingService svc) => _svc = svc;

    /// <summary>Analiza una URL y retorna la prediccion del modelo.</summary>
    [HttpPost("analyze")]
    public async Task<ActionResult<AnalysisResponse>> Analyze([FromBody] AnalyzeRequest req)
    {
        if (!ModelState.IsValid)
            return BadRequest(ModelState);

        var result = await _svc.AnalyzeUrlAsync(req.Url);
        return Ok(result);
    }

    /// <summary>Retorna el historial de análisis.</summary>
    [HttpGet("history")]
    public async Task<ActionResult<List<Analysis>>> History([FromQuery] int limit = 50)
    {
        var history = await _svc.GetHistoryAsync(limit);
        return Ok(history);
    }
}

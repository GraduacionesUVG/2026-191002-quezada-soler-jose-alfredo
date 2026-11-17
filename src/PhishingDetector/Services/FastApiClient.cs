using System.Net.Http.Json;
using PhishingDetector.Models;

namespace PhishingDetector.Services;

public class FastApiClient
{
    private readonly HttpClient _http;
    private readonly ILogger<FastApiClient> _log;

    public FastApiClient(HttpClient http, ILogger<FastApiClient> log)
    {
        _http = http;
        _log = log;
    }

    public async Task<PhishingResult> AnalyzeAsync(string url)
    {
        var response = await _http.PostAsJsonAsync("/analyze", new { url });
        response.EnsureSuccessStatusCode();
        var result = await response.Content.ReadFromJsonAsync<PhishingResult>()
            ?? throw new InvalidOperationException("Respuesta vacía del servicio IA.");
        return result;
    }

    public async Task<bool> IsHealthyAsync()
    {
        try
        {
            var r = await _http.GetAsync("/health");
            return r.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }
}

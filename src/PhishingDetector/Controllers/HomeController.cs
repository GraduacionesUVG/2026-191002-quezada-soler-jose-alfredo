using Microsoft.AspNetCore.Mvc;
using PhishingDetector.Models;
using PhishingDetector.Services;

namespace PhishingDetector.Controllers;

public class HomeController : Controller
{
    private readonly IPhishingService _svc;
    private readonly ILogger<HomeController> _log;

    public HomeController(IPhishingService svc, ILogger<HomeController> log)
    {
        _svc = svc;
        _log = log;
    }

    public IActionResult Index() => View(new AnalyzeRequest());

    [HttpPost]
    public async Task<IActionResult> Analyze(AnalyzeRequest request)
    {
        if (!ModelState.IsValid)
            return View("Index");

        try
        {
            var result = await _svc.AnalyzeUrlAsync(request.Url);
            return View("Result", result);
        }
        catch (Exception ex)
        {
            _log.LogError(ex, "Error al analizar URL {Url}", request.Url);
            ModelState.AddModelError(string.Empty,
                "No se pudo conectar al servicio de IA. Asegúrate de que FastAPI está corriendo en el puerto 8000.");
            return View("Index");
        }
    }

    public IActionResult Error()
    {
        return View();
    }
}

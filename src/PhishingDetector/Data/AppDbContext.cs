using Microsoft.EntityFrameworkCore;
using PhishingDetector.Models;

namespace PhishingDetector.Data;

public class AppDbContext : DbContext
{
    public AppDbContext(DbContextOptions<AppDbContext> options) : base(options) { }

    public DbSet<Analysis> Analyses => Set<Analysis>();

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder.Entity<Analysis>(e =>
        {
            e.ToTable("analyses");
            e.HasKey(x => x.Id);
            e.Property(x => x.Url).IsRequired().HasMaxLength(2048);
            e.Property(x => x.Result).IsRequired().HasMaxLength(20);
            e.Property(x => x.RiskLevel).HasMaxLength(10);
        });
    }
}

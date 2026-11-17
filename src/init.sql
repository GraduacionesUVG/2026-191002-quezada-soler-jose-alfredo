-- Schema inicial (EnsureCreated lo aplica automáticamente; este archivo es
-- para uso directo con Docker o cuando se crea la DB manualmente)

CREATE TABLE IF NOT EXISTS analyses (
    "Id"          SERIAL PRIMARY KEY,
    "Url"         VARCHAR(2048)    NOT NULL,
    "Result"      VARCHAR(20)      NOT NULL,
    "Probability" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "RiskLevel"   VARCHAR(10),
    "CreatedAt"   TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses ("CreatedAt" DESC);

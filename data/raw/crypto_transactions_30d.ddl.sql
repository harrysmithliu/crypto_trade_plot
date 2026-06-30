-- PostgreSQL DDL for data/raw/crypto_transactions_30d.csv
-- This schema is based on the current mock dataset and validation rules.

CREATE TABLE IF NOT EXISTS crypto_transactions_30d (
    tx_id VARCHAR(32) PRIMARY KEY,
    order_id VARCHAR(32) NOT NULL UNIQUE,
    user_id VARCHAR(16) NOT NULL,
    "timestamp" TIMESTAMPTZ NOT NULL,
    merchant_id VARCHAR(16) NOT NULL,
    merchant_name VARCHAR(128) NOT NULL,
    crypto_asset VARCHAR(8) NOT NULL,
    fiat_currency CHAR(3) NOT NULL,
    fiat_volume_cad NUMERIC(18, 2) NOT NULL,
    tx_status VARCHAR(16) NOT NULL,
    payment_channel VARCHAR(32) NOT NULL,
    decline_reason VARCHAR(64),
    processor_code VARCHAR(32) NOT NULL,
    flat_fee_cad NUMERIC(18, 2) NOT NULL DEFAULT 0,
    spread_income_cad NUMERIC(18, 2) NOT NULL DEFAULT 0,
    flow_direction VARCHAR(16) NOT NULL,
    risk_score NUMERIC(5, 2) NOT NULL,
    aml_flag BOOLEAN NOT NULL,
    ip_country CHAR(2) NOT NULL,
    velocity_1h INTEGER NOT NULL,
    is_high_risk BOOLEAN NOT NULL,
    order_created_at TIMESTAMPTZ NOT NULL,
    channel_selected_at TIMESTAMPTZ,
    kyc_passed_at TIMESTAMPTZ,
    payment_completed_at TIMESTAMPTZ,
    settled_at TIMESTAMPTZ,
    provider_amount_cad NUMERIC(18, 2) NOT NULL,
    ledger_amount_cad NUMERIC(18, 2) NOT NULL,
    recon_delta_cad NUMERIC(18, 2) NOT NULL DEFAULT 0,
    recon_status VARCHAR(16) NOT NULL,
    latency_ms INTEGER NOT NULL,
    settlement_latency_min INTEGER,
    CONSTRAINT chk_crypto_asset
        CHECK (crypto_asset IN ('USDC', 'USDT', 'BTC', 'ETH')),
    CONSTRAINT chk_fiat_currency
        CHECK (fiat_currency IN ('CAD', 'USD')),
    CONSTRAINT chk_tx_status
        CHECK (tx_status IN ('Completed', 'Pending', 'Failed')),
    CONSTRAINT chk_payment_channel
        CHECK (payment_channel IN (
            'Interac e-Transfer',
            'Wire Transfer',
            'Crypto Network',
            'Card'
        )),
    CONSTRAINT chk_flow_direction
        CHECK (flow_direction IN ('Inflow', 'Outflow')),
    CONSTRAINT chk_ip_country
        CHECK (ip_country IN ('CA', 'US', 'GB', 'HK', 'SG', 'NG')),
    CONSTRAINT chk_recon_status
        CHECK (recon_status IN ('Matched', 'Mismatch')),
    CONSTRAINT chk_fiat_volume_cad_nonnegative
        CHECK (fiat_volume_cad >= 0),
    CONSTRAINT chk_flat_fee_cad_nonnegative
        CHECK (flat_fee_cad >= 0),
    CONSTRAINT chk_spread_income_cad_nonnegative
        CHECK (spread_income_cad >= 0),
    CONSTRAINT chk_risk_score_range
        CHECK (risk_score >= 0 AND risk_score <= 100),
    CONSTRAINT chk_velocity_1h_nonnegative
        CHECK (velocity_1h >= 0),
    CONSTRAINT chk_latency_ms_nonnegative
        CHECK (latency_ms >= 0),
    CONSTRAINT chk_settlement_latency_min_nonnegative
        CHECK (settlement_latency_min IS NULL OR settlement_latency_min >= 0)
);

CREATE INDEX IF NOT EXISTS idx_crypto_transactions_30d_timestamp
    ON crypto_transactions_30d ("timestamp");

CREATE INDEX IF NOT EXISTS idx_crypto_transactions_30d_merchant_id
    ON crypto_transactions_30d (merchant_id);

CREATE INDEX IF NOT EXISTS idx_crypto_transactions_30d_tx_status
    ON crypto_transactions_30d (tx_status);

CREATE INDEX IF NOT EXISTS idx_crypto_transactions_30d_recon_status
    ON crypto_transactions_30d (recon_status);

COMMENT ON TABLE crypto_transactions_30d IS
    'Mock 30-day crypto payment transaction dataset used for analytics, dashboarding, and reconciliation monitoring.';

COMMENT ON COLUMN crypto_transactions_30d.tx_id IS
    'Unique transaction identifier for each payment event.';
COMMENT ON COLUMN crypto_transactions_30d.order_id IS
    'Order identifier linked to the payment transaction.';
COMMENT ON COLUMN crypto_transactions_30d.user_id IS
    'Internal user identifier associated with the order.';
COMMENT ON COLUMN crypto_transactions_30d."timestamp" IS
    'Primary transaction event timestamp stored in UTC.';
COMMENT ON COLUMN crypto_transactions_30d.merchant_id IS
    'Internal merchant identifier.';
COMMENT ON COLUMN crypto_transactions_30d.merchant_name IS
    'Display name of the merchant.';
COMMENT ON COLUMN crypto_transactions_30d.crypto_asset IS
    'Crypto asset used in the transaction, such as USDC, USDT, BTC, or ETH.';
COMMENT ON COLUMN crypto_transactions_30d.fiat_currency IS
    'Quoted fiat currency context for the transaction, such as CAD or USD.';
COMMENT ON COLUMN crypto_transactions_30d.fiat_volume_cad IS
    'Transaction amount normalized into CAD for reporting and aggregation.';
COMMENT ON COLUMN crypto_transactions_30d.tx_status IS
    'Lifecycle status of the transaction: Completed, Pending, or Failed.';
COMMENT ON COLUMN crypto_transactions_30d.payment_channel IS
    'Payment rail or processing channel used by the transaction.';
COMMENT ON COLUMN crypto_transactions_30d.decline_reason IS
    'Failure reason when the transaction is declined; null or empty for non-failed records.';
COMMENT ON COLUMN crypto_transactions_30d.processor_code IS
    'Provider or internal processor result code for the transaction outcome.';
COMMENT ON COLUMN crypto_transactions_30d.flat_fee_cad IS
    'Flat fee revenue recognized by the platform in CAD.';
COMMENT ON COLUMN crypto_transactions_30d.spread_income_cad IS
    'Spread-based revenue recognized by the platform in CAD.';
COMMENT ON COLUMN crypto_transactions_30d.flow_direction IS
    'Fund movement direction, such as Inflow or Outflow.';
COMMENT ON COLUMN crypto_transactions_30d.risk_score IS
    'Risk score on a 0 to 100 scale used for fraud and AML review.';
COMMENT ON COLUMN crypto_transactions_30d.aml_flag IS
    'Boolean flag indicating whether AML review conditions were triggered.';
COMMENT ON COLUMN crypto_transactions_30d.ip_country IS
    'Country code inferred from the originating IP address.';
COMMENT ON COLUMN crypto_transactions_30d.velocity_1h IS
    'Count of recent user activities or transactions within the last hour.';
COMMENT ON COLUMN crypto_transactions_30d.is_high_risk IS
    'Boolean flag indicating high-risk classification based on risk rules.';
COMMENT ON COLUMN crypto_transactions_30d.order_created_at IS
    'UTC timestamp when the order was initially created.';
COMMENT ON COLUMN crypto_transactions_30d.channel_selected_at IS
    'UTC timestamp when the user selected a payment channel.';
COMMENT ON COLUMN crypto_transactions_30d.kyc_passed_at IS
    'UTC timestamp when KYC review was completed successfully.';
COMMENT ON COLUMN crypto_transactions_30d.payment_completed_at IS
    'UTC timestamp when payment completion was confirmed.';
COMMENT ON COLUMN crypto_transactions_30d.settled_at IS
    'UTC timestamp when the transaction was considered settled.';
COMMENT ON COLUMN crypto_transactions_30d.provider_amount_cad IS
    'External provider-side settlement amount in CAD.';
COMMENT ON COLUMN crypto_transactions_30d.ledger_amount_cad IS
    'Internal ledger-side settlement amount in CAD.';
COMMENT ON COLUMN crypto_transactions_30d.recon_delta_cad IS
    'Difference in CAD between provider and ledger settlement amounts.';
COMMENT ON COLUMN crypto_transactions_30d.recon_status IS
    'Reconciliation result, typically Matched or Mismatch.';
COMMENT ON COLUMN crypto_transactions_30d.latency_ms IS
    'Processing latency in milliseconds for the transaction event.';
COMMENT ON COLUMN crypto_transactions_30d.settlement_latency_min IS
    'Elapsed time in minutes from payment completion to settlement.';

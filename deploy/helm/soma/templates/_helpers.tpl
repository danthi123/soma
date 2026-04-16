{{/*
Expand the name of the chart.
*/}}
{{- define "soma.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited
to this (by the DNS naming spec). If release name contains chart name it
will be used as a full name.
*/}}
{{- define "soma.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Create chart name + version as used by the chart label.
*/}}
{{- define "soma.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Common labels — standard Kubernetes recommended labels.
*/}}
{{- define "soma.labels" -}}
helm.sh/chart: {{ include "soma.chart" . }}
{{ include "soma.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: soma
{{- end -}}

{{/*
Selector labels — the subset used by Service/StatefulSet selectors.
Must be stable across upgrades (never include chart version).
*/}}
{{- define "soma.selectorLabels" -}}
app.kubernetes.io/name: {{ include "soma.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
Create the name of the service account to use.
*/}}
{{- define "soma.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "soma.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/*
Name of the Secret that holds SOMA_API_KEY. When `api.existingSecret`
is set we honor that; otherwise we use a deterministic `<fullname>-api`
name so `lookup` can find the existing value on upgrade.
*/}}
{{- define "soma.apiSecretName" -}}
{{- if .Values.api.existingSecret -}}
{{- .Values.api.existingSecret -}}
{{- else -}}
{{- printf "%s-api" (include "soma.fullname" .) -}}
{{- end -}}
{{- end -}}

{{/*
Key inside the API Secret.
*/}}
{{- define "soma.apiSecretKey" -}}
{{- default "SOMA_API_KEY" .Values.api.existingSecretKey -}}
{{- end -}}

{{/*
Idempotent API key generation.

Order of precedence:
  1. If `api.existingSecret` is set, the helper is never called (secret
     template is skipped entirely; Pod references `existingSecret`).
  2. Else, look up an existing Secret in the release namespace — preserve
     its value across `helm upgrade` so keys survive.
  3. Else, mint a new 32-byte hex string on first install.

`lookup` returns `{}` in `helm template` / dry-run; we fall back to a
generated key so rendering is deterministic in CI. Real clusters see
the preserved value on upgrade because `lookup` returns the live Secret.
*/}}
{{- define "soma.apiKey" -}}
{{- $name := include "soma.apiSecretName" . -}}
{{- $ns := .Release.Namespace -}}
{{- $existing := (lookup "v1" "Secret" $ns $name) -}}
{{- $key := include "soma.apiSecretKey" . -}}
{{- if and $existing $existing.data (index $existing.data $key) -}}
{{- index $existing.data $key | b64dec -}}
{{- else -}}
{{- randAlphaNum 32 -}}
{{- end -}}
{{- end -}}

{{/*
Validate networking values — fail fast if both Ingress and HTTPRoute
are enabled. Called at the top of ingress.yaml and httproute.yaml.
*/}}
{{- define "soma.validateNetworking" -}}
{{- if and .Values.ingress.enabled .Values.httpRoute.enabled -}}
{{- fail "Enable exactly one of ingress.enabled or httpRoute.enabled (mutually exclusive)." -}}
{{- end -}}
{{- end -}}

{{/*
Validate replica lock — defense-in-depth behind `values.schema.json`.
WAL is single-writer; enforce again at render time in case schema is
bypassed (older Helm, external tooling).
*/}}
{{- define "soma.validateReplicas" -}}
{{- if gt (int .Values.replicaCount) 1 -}}
{{- fail "replicaCount must be 1 — WAL is single-writer. Unlocked when Phase 6 ships external VectorBackend." -}}
{{- end -}}
{{- end -}}

{{/*
Headless service name — used by the StatefulSet for stable pod DNS.
*/}}
{{- define "soma.headlessServiceName" -}}
{{- printf "%s-headless" (include "soma.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Resolve the image reference. When `image.tag` is empty, fall back to
the Chart's `appVersion` so every chart version pins a default image.
*/}}
{{- define "soma.image" -}}
{{- $tag := default .Chart.AppVersion .Values.image.tag -}}
{{- printf "%s:%s" .Values.image.repository $tag -}}
{{- end -}}

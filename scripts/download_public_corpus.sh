#!/usr/bin/env bash

set -u

start=1
count=25
delay_ms=250
while [ "$#" -gt 0 ]; do
  case "$1" in
    --start) start="$2"; shift 2 ;;
    --count) count="$2"; shift 2 ;;
    --delay-ms) delay_ms="$2"; shift 2 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

repo_root=$(cd "$(dirname "$0")/.." && pwd)
raw_dir="$repo_root/data/raw/raib"
manifest="$repo_root/data/catalog/raib_manifest.csv"
work_dir=$(mktemp -d "${TMPDIR:-/tmp}/raib-segment.XXXXXX")
trap 'rm -rf "$work_dir"' EXIT

mkdir -p "$raw_dir" "$(dirname "$manifest")"
ua='railway-agent-safety-security-research/0.1 (public research corpus)'
list_file="$repo_root/data/catalog/raib_report_urls.txt"

if [ ! -s "$list_file" ]; then
  list_file_tmp="$work_dir/report_urls"
  for page in $(seq 1 20); do
  list_url='https://www.gov.uk/raib-reports'
  [ "$page" -gt 1 ] && list_url="${list_url}?page=${page}"
    curl -L --fail --silent --show-error --connect-timeout 15 --max-time 45 \
      -A "$ua" "$list_url" -o "$work_dir/list-$page.html" || continue
    rg -o 'href="/raib-reports/[^"]+"' "$work_dir/list-$page.html" \
      | sed 's/^href="//; s/"$//' \
      | sed '/\/email-signup$/d; /\/raib-reports\/page/d' \
      | sed 's#^#https://www.gov.uk#' >> "$list_file_tmp"
    sleep 0.25
  done
  sort -u "$list_file_tmp" > "$list_file"
fi

total=$(wc -l < "$list_file")
end=$((start + count - 1))
[ "$end" -gt "$total" ] && end="$total"
[ ! -f "$manifest" ] && printf 'report_url,pdf_url,file,status,error\n' > "$manifest"

for index in $(seq "$start" "$end"); do
  report_url=$(sed -n "${index}p" "$list_file")
  [ -n "$report_url" ] || continue
  if ! curl -L --fail --silent --show-error --connect-timeout 15 --max-time 45 \
    -A "$ua" "$report_url" -o "$work_dir/report.html"; then
    printf '%s,,,error,report_page_fetch_failed\n' "$report_url" >> "$manifest"
    printf '[%s/%s] report page failed\n' "$index" "$total"
    continue
  fi

  pdf_url=$(rg -o 'href="https://assets\.publishing\.service\.gov\.uk/[^"]+\.pdf"' \
    "$work_dir/report.html" | head -1 | sed 's/^href="//; s/"$//' | sed 's/&amp;/\&/g')
  if [ -z "$pdf_url" ]; then
    printf '%s,,,no-pdf,\n' "$report_url" >> "$manifest"
    printf '[%s/%s] no PDF\n' "$index" "$total"
    continue
  fi

  file_name=$(basename "${pdf_url%%\?*}" | sed 's/[^a-zA-Z0-9._-]/_/g')
  target="$raw_dir/$file_name"
  status=already_present
  if [ ! -s "$target" ]; then
    if curl -L --fail --silent --show-error --connect-timeout 15 --max-time 90 \
      -A "$ua" "$pdf_url" -o "$target"; then
      status=downloaded
    else
      status=error
      printf '%s,,,error,pdf_download_failed\n' "$report_url" >> "$manifest"
    fi
  fi
  if [ "$status" != error ]; then
    printf '%s,%s,%s,%s,\n' "$report_url" "$pdf_url" "$file_name" "$status" >> "$manifest"
  fi
  printf '[%s/%s] %s (%s)\n' "$index" "$total" "$file_name" "$status"
  sleep "0.$(printf '%03d' "$delay_ms")"
done

printf 'Segment finished: %s-%s of %s; PDF files: ' "$start" "$end" "$total"
find "$raw_dir" -maxdepth 1 -type f -name '*.pdf' | wc -l
